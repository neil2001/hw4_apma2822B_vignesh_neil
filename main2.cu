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

int numStreams = 3;

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
            // printf("%ld, %g, %g, %g\n", (row * n) + i, rows[offset + i], vec[i], sum);
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

    int M = 300;
    int N = 1000;

    int rowsPerBlock = (M / numStreams);

    double mat_h[M*N] = {0};
    double mat_h0[rowsPerBlock*N] = {0};
    double mat_h1[rowsPerBlock*N] = {0};
    double mat_h2[rowsPerBlock*N] = {0};

    double vec_h[N] = {0};
    double res_h0[rowsPerBlock] = {0};
    double res_h1[rowsPerBlock] = {0};
    double res_h2[rowsPerBlock] = {0};

    instantiateMatVec(M, N, mat_h, vec_h);
    instantiateMatVec(rowsPerBlock, N, mat_h0, vec_h);
    instantiateMatVec(rowsPerBlock, N, mat_h1, vec_h);
    instantiateMatVec(rowsPerBlock, N, mat_h2, vec_h);

    // instantiateMatVec(M, N, mat_h, vec_h);

    double *mat_d0;
    double *mat_d1;
    double *mat_d2;

    double *res_d0;
    double *res_d1;
    double *res_d2;

    struct timeval startTime;
    struct timeval endTime;

    dim3 nthreads(rowsPerBlock, 1, 1);
    dim3 nblocks (1, 1, 1); 
    
    gettimeofday(&startTime, nullptr);  

    cudaStream_t stream0, stream1, stream2;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

        // allocate memory on device
    cudaMalloc( (void**) &mat_d0, sizeof(double)*rowsPerBlock*N);
    cudaMalloc( (void**) &mat_d1, sizeof(double)*rowsPerBlock*N);
    cudaMalloc( (void**) &mat_d2, sizeof(double)*rowsPerBlock*N);

    cudaHostAlloc( (void**) &vec_h, sizeof(double)*N, cudaHostAllocDefault);
    instantiateMatVec(rowsPerBlock, N, mat_h2, vec_h);

    cudaMalloc( (void**) &res_d0, sizeof(double)*rowsPerBlock);
    cudaMalloc( (void**) &res_d1, sizeof(double)*rowsPerBlock);
    cudaMalloc( (void**) &res_d2, sizeof(double)*rowsPerBlock);

    // cudaMemcpy(vec_d, vec_h, sizeof(double)*N,cudaMemcpyHostToDevice);

    // for (int i = 0; i < 10; i++) {
    cudaMemcpyAsync(mat_d0, mat_h0, sizeof(double)*rowsPerBlock*N, cudaMemcpyHostToDevice, stream0);
    cudaMemcpyAsync(mat_d1, mat_h1, sizeof(double)*rowsPerBlock*N, cudaMemcpyHostToDevice, stream1);
    cudaMemcpyAsync(mat_d2, mat_h2, sizeof(double)*rowsPerBlock*N, cudaMemcpyHostToDevice, stream2);

    matVecKernel<<<nblocks, nthreads, 0, stream0>>>(M, N, mat_d0, vec_h, res_d0);
    matVecKernel<<<nblocks, nthreads, 0, stream1>>>(M, N, mat_d1, vec_h, res_d1);
    matVecKernel<<<nblocks, nthreads, 0, stream2>>>(M, N, mat_d2, vec_h, res_d2);

    cudaMemcpyAsync(res_h0, res_d0, sizeof(double)*rowsPerBlock, cudaMemcpyDeviceToHost, stream0);
    cudaMemcpyAsync(res_h1, res_d1, sizeof(double)*rowsPerBlock, cudaMemcpyDeviceToHost, stream1);
    cudaMemcpyAsync(res_h2, res_d2, sizeof(double)*rowsPerBlock, cudaMemcpyDeviceToHost, stream2);
        // cudaMemcpyAsync(&res_h[rowsPerBlock], &res_d[rowsPerBlock], sizeof(double)*min(rowsPerBlock, M - rowsPerBlock), cudaMemcpyDeviceToHost, stream1);
    // }

    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    gettimeofday(&endTime, nullptr);

    int microseconds = (endTime.tv_sec - startTime.tv_sec) * 1000000 + (endTime.tv_usec - startTime.tv_usec);
    std::cout << "creating streams took " << microseconds << " microseconds" << std::endl;

    // double expected[M] = {0};
    // for (int i = 0; i < M; i++) {
    //     double sum = 0.0;
    //     int offset = i * N;
    //     for (int j = 0; j < N; j++) {
    //         sum += mat_h[offset+j] * vec_h[j];
    //     }
    //     expected[i] = sum;
    // }

    // for (int i = 0; i < rowsPerBlock; i++) {
    //     if (abs(expected[i] - res_h0[i]) > 0.0001) {
    //         printf("DIFF FOUND: expected: %g, actual: %g", expected[i], res_h0[i]);
    //     } else if (i % 100 == 0) {
    //         printf("no diff, found %g \n", expected[i]);
    //     }
    // }
    // for (int i = 0; i < rowsPerBlock; i++) {
    //     if (abs(expected[i+rowsPerBlock] - res_h1[i]) > 0.0001) {
    //         printf("DIFF FOUND: expected: %g, actual: %g", expected[i+rowsPerBlock], res_h1[i]);
    //     } else if (i % 100 == 0) {
    //         printf("no diff, found %g \n", expected[i+rowsPerBlock]);
    //     }
    // }

    cudaFree(res_d0);
    cudaFree(res_d1);
    cudaFree(res_d2);

    // cudaFree(vec_d);
    cudaFreeHost(vec_h);

    cudaFree(mat_d0);
    cudaFree(mat_d1);
    cudaFree(mat_d2);
}
