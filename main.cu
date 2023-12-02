#include <cuda.h>
#include <stdio.h>

#define N 1024
#define __WARP_SIZE__ 32
#define FULL_MASK 0xffffffff

int main() {
    int ncuda_devices = 0;
    cudaGetDeviceCount(&ncuda_devices);
    printf("ncuda_devices = %d\n",ncuda_devices);

    if (ncuda_devices == 0) {
        fprintf(stderr,"NO CUDA DEVICES EXITING\n");
        return 0;
    }
    cudaSetDevice(0);

    int M = 10;
    int N = 10;

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
}