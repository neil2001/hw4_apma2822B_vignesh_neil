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

const int nwarps = 32;

int numStreams = 4;

// randomization
std::random_device rd;
std::mt19937 gen(rd()); // Mersenne Twister 19937 generator
std::uniform_real_distribution<double> distribution(1.0, 100.0);

__global__ void matVecKernel(int m, int n, double *rows, double *vec, double *res, int streamNun, int rowsPerStream) {
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m) {
        // printf("Row: %d\n", (int) row);
        double sum = 0.0;
        int offset = row * n;
        for (int i = 0; i < n; i++) {
            // printf("r,c = (%d, %d)\n", (int) row, i);
            sum += rows[offset + i] * vec[i];
            // printf("%ld, %g, %g, %g\n", (row * n) + i, rows[offset + i], vec[i], sum);
        }
        res[row] = sum;

        __syncthreads();
    }
}

__global__ void matVecMultiWarpKernel(int rows, int cols, double *mat, double *vec, double *res, int streamNum, int rowsPerStream) {
    size_t row = blockIdx.x + streamNum * rowsPerStream;

    int nwarps = blockDim.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    if (row >= rows) {
        return;
    }

    double sum = 0.0;
    int row_offset = blockIdx.x * cols;
    for (int i = lane + (WARP_SIZE * warp_id); i < cols; i += WARP_SIZE * nwarps) { // TODO: figure out indexing
        if (i < cols) {
            sum += mat[row_offset + i] * vec[i];
        }
    }

    __syncwarp();

    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(FULL_MASK, sum, offset);
    }

    __shared__ double s_mem[1024/WARP_SIZE];
    if (lane == 0) {
        s_mem[warp_id] = sum;
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        double f_sum = 0.0;
        for (int i = 0; i < nwarps; i++) {
            f_sum += s_mem[i];
        }
        res[blockIdx.x] = f_sum;
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

    double *mat_h;
    double *vec_h;
    double *res_h;

    cudaHostAlloc( (void**)&mat_h, M*N*sizeof(double), cudaHostAllocWriteCombined);
    cudaHostAlloc( (void**)&vec_h, N*sizeof(double), cudaHostAllocWriteCombined);
    cudaHostAlloc( (void**)&res_h, M*sizeof(double), cudaHostAllocWriteCombined);

    instantiateMatVec(M, N, mat_h, vec_h);

    double *mat_d;
    double *vec_d;
    double *res_d;

    int rowsPerBlock = (M + numStreams - 1) / numStreams;

    // allocate memory on device
    cudaMalloc( (void**) &mat_d, sizeof(double)*M*N);
    cudaMalloc( (void**) &vec_d, sizeof(double)*N);
    cudaMalloc( (void**) &res_d, sizeof(double)*M);

    cudaMemcpy(vec_d, vec_h, sizeof(double)*N,cudaMemcpyHostToDevice);

    struct timeval startTime;
    struct timeval endTime;


    cudaStream_t streams[numStreams];


    for (int i=0; i<numStreams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // dim3 nthreads(256, 1, 1); // threads per block NOTE NOT MORE THAN 1024
    // dim3 nblocks ((rowsPerBlock + nthreads.x - 1)/nthreads.x, 1, 1); // blocks per grid -> should be 1
    
    dim3 nthreads(WARP_SIZE * nwarps,1,1);
	dim3 nblocks(rowsPerBlock,1,1);
    gettimeofday(&startTime, nullptr);  

    for (int i=0; i<numStreams; i++) {
        cudaMemcpyAsync(&mat_d[i*rowsPerBlock*N], &mat_h[i*rowsPerBlock*N], sizeof(double)*rowsPerBlock*N, cudaMemcpyHostToDevice, streams[i]);
        matVecMultiWarpKernel<<<nblocks, nthreads, 0, streams[i]>>>(M, N, &mat_d[i*rowsPerBlock*N], vec_d, &res_d[i*rowsPerBlock], i, rowsPerBlock);
        cudaMemcpyAsync(&res_h[i*rowsPerBlock], &res_d[i*rowsPerBlock], sizeof(double)*rowsPerBlock, cudaMemcpyDeviceToHost, streams[i]);
    }

    cudaDeviceSynchronize();

    for (int i = 0; i < numStreams; i++) {
        cudaStreamDestroy(streams[i]);
    }

    gettimeofday(&endTime, nullptr);

    int microseconds = (endTime.tv_sec - startTime.tv_sec) * 1000000 + (endTime.tv_usec - startTime.tv_usec);
    std::cout << "executing streams took " << microseconds << " microseconds" << std::endl;

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
        }
    }

    cudaFree(res_d);
    cudaFree(vec_d);
    cudaFree(mat_d);

    cudaFreeHost(mat_h);
    cudaFreeHost(vec_h);
    cudaFreeHost(res_h);
}