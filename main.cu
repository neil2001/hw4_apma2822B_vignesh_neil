#include <cuda.h>
#include <stdio.h>
#include <sys/time.h>
#include <random>
#include <fstream>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

using namespace std;

// Matrix Vector Multiplication with GPU Streams
#define WARP_SIZE 32
#define FULL_MASK 0xffffffff
#define NWARPS 32
#define ROWS_PER_BLOCK 4

int numStreams = 8;

// randomization
std::random_device rd;
std::mt19937 gen(rd()); // Mersenne Twister 19937 generator
std::uniform_real_distribution<double> distribution(1.0, 100.0);

__global__ void matVecKernel(int m, int n, double *rows, double *vec, double *res) {
    size_t row = threadIdx.x;
    
    if (row < m) {
        // printf("Row: %d\n", (int) row);
        double sum = 0.0;
        int offset = row * n;
        for (int i = 0; i < n; i++) {
            // printf("r,c = (%d, %d)\n", (int) row, i);
            sum += rows[offset + i] * vec[i];
            // printf("%d, %g, %g\n", (row * cols) + i, vector[i], sum);
        }
        res[row] = sum;
    }
}

void instantiateMatVec(int m, int n, double *mat, double *vec) {
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
        mat[n * i + j] = distribution(gen);  // Generate a random double value and store it in the matrix
    }
  }
  for (int i = 0; i < n; i++) {
    vec[i] = distribution(gen);
  }
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Concurrent Kernels: " << (prop.concurrentKernels ? "Yes" : "No") << std::endl;

    int ncuda_devices = 0;
    cudaGetDeviceCount(&ncuda_devices);
    printf("ncuda_devices = %d\n",ncuda_devices);

    if (ncuda_devices == 0) {
        fprintf(stderr,"NO CUDA DEVICES EXITING\n");
        return 0;
    }
    cudaSetDevice(0);


    int M = 1000;
    int N = 10000;

    double mat_h[M*N] = {0};
    double vec_h[N] = {0};
    double res_h[M] = {0};

    instantiateMatVec(M, N, mat_h, vec_h);

    double *mat_d;
    double *vec_d;
    double *res_d;

    // allocate memory on device
    cudaMalloc( (void**) &mat_d, sizeof(double)*M*N);
    cudaMalloc( (void**) &vec_d, sizeof(double)*N);
    cudaMalloc( (void**) &res_d, sizeof(double)*M);

    cudaMemcpy(vec_d, vec_h, sizeof(double)*N,cudaMemcpyHostToDevice);

    struct timeval startTime;
    struct timeval endTime;


    cudaStream_t streams[numStreams];

    int rowsPerBlock = (M / numStreams) + 1;

    for (int i=0; i<numStreams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    dim3 nthreads(rowsPerBlock, 1, 1); // threads per block NOTE NOT MORE THAN 1024
    // 30 rows per block
    // 1 thread per row

    dim3 nblocks (1, 1, 1); // blocks per grid -> should be 1
    // dim3 nblocks ((rowsPerBlock + nthreads.x-1)/nthreads.x, 1, 1); // blocks per grid -> should be 1
    gettimeofday(&startTime, nullptr);  

    for (int i=0; i<numStreams; i++) {
        // copy H2D
        int numToCpy = min(N*M - i*rowsPerBlock*N, rowsPerBlock*N);
        cudaMemcpyAsync(&mat_d[i*rowsPerBlock*N], &mat_h[i*rowsPerBlock*N], sizeof(double)*numToCpy, cudaMemcpyHostToDevice, streams[i]);
        // run / kernel func
        matVecKernel<<<nblocks, nthreads, 0, streams[i]>>>(M, N, &mat_d[i*rowsPerBlock*N], vec_d, &res_d[i*rowsPerBlock]);
        // cudaMemcpyAsync(mat_h[], mat_d[], sizeof(double)*, cudaMemcpyDeviceToDevice, streams[i]);
        cudaMemcpyAsync(&res_h[i*rowsPerBlock], &res_d[i*rowsPerBlock], sizeof(double)*min(rowsPerBlock, M - i * rowsPerBlock), cudaMemcpyDeviceToHost, streams[i]);
        // copy D2H
    }
    
    for (int i = 0; i < numStreams; i++) {
        cudaStreamSynchronize(streams[i]);
    }
    for (int i = 0; i < numStreams; i++) {
        cudaStreamDestroy(streams[i]);
    }

    cudaDeviceSynchronize();

    gettimeofday(&endTime, nullptr);

    int microseconds = (endTime.tv_sec - startTime.tv_sec) * 1000000 + (endTime.tv_usec - startTime.tv_usec);
    std::cout << "creating streams took " << microseconds << " microseconds" << std::endl;

    double expected[M] = {0};
    for (int i = 0; i < M; i++) {
        double sum = 0.0;
        int offset = i * N;
        for (int j = 0; j < N; j++) {
            sum += mat_h[offset+j] * vec_h[j];
        }
        expected[i] = sum;
    }

    for (int i = 0; i < M; i++) {
        if (abs(expected[i] - res_h[i]) > 0.0001) {
            printf("DIFF FOUND: expected: %g, actual: %g", expected[i], res_h[i]);
        } else if (i % 100 == 0) {
            printf("no diff, found %g \n", expected[i]);
        }
    }
    cudaFree(res_d);
    cudaFree(vec_d);
    cudaFree(mat_d);
}