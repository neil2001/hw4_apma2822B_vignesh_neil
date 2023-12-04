#include <cuda.h>
#include <stdio.h>
#include <sys/time.h>
#include <random>
#include <fstream>
#include <iostream>
#include <cmath>

using namespace std;

// Matrix Vector Multiplication with GPUs
#define WARP_SIZE 32
#define FULL_MASK 0xffffffff
#define NWARPS 32
#define ROWS_PER_BLOCK 4

const int nwarps = 32;

enum Experiment {
    NO_WARP,
    MULTI_WARP
};

typedef void (*MatVecMulFunc)(double *matrix, double *vector, double *result, int rows, int cols);

// randomization
std::random_device rd;
std::mt19937 gen(rd()); // Mersenne Twister 19937 generator
std::uniform_real_distribution<double> distribution(1.0, 100.0);

__global__ void matVecMultiWarp(int rows, int cols, double *mat, double *vec, double *res, int streamNum, int rowsPerStream) {
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

__global__
void matVecMulNoWarp(int m, int n, double *rows, double *vec, double *res) {
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

double getFlopRate(int flops, int ms)
{
    double flopsms = ((double)flops) / ((double)ms);
    return flopsms / 1000000.0;
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

void generateLatexTable(int* N, int* M, int* executionTime, double* flopRate, int* numStreams, int size, std::string tableTitle, std::string output) {
    std::ofstream myfile;
    myfile.open(output);
    myfile << "\\begin{table}[htbp]\n";
    myfile << "  \\centering\n";
    myfile << "  \\caption{" << tableTitle << "}\n";
    myfile << "  \\begin{tabular}{|c|c|c|c|c|}\n";
    myfile << "    \\hline\n";
    myfile << "    \\multirow{2}{*}{N} & \\multirow{2}{*}{M} & \\multirow{2}{*}{Streams} & \\multicolumn{2}{c|}{Performance Metrics} \\\\\n";
    myfile << "    \\cline{4-5}\n";
    myfile << "    & & & Execution Time (ms) & Flop Rate (TFLOP/s) \\\\\n";
    myfile << "    \\hline\n";


    myfile << "    " << N[i] << " & " << M[i] << " & " << numStreams[i] << " & " << executionTime[i] << " & " << flopRate[i] << " \\\\\n";
    for (int i = 1; i < size; i++) {
        myfile << "    " << " & & " << numStreams[i] << " & " << executionTime[i] << " & " << flopRate[i] << " \\\\\n";
    }

    myfile << "    \\hline\n";
    myfile << "  \\end{tabular}\n";
    myfile << "\\end{table}\n";
    myfile.close();
}

void runExperiment(Experiment e, std::string output) {

    // int rowDims[] = {10, 10, 10, 10, 100, 100, 100, 100, 1000, 1000, 1000, 1000, 10000, 10000, 10000};
    // int colDims[] = {10, 100, 1000, 10000, 10, 100, 1000, 10000, 10, 100, 1000, 10000, 10, 100, 1000};
    int rds[] = {1000, 1500}; // , 2000, 2500, 5000};
    int cds[] = {1000, 1500}; //, 2000, 2500, 5000};

    int dims = (sizeof(rds) / sizeof(rds[0]));
    int rowDims[dims*8];
    int colDims[dims*8];

    for (int i=0; i<dims*8; i++) {
        rowDims[i] = rds[i/8];
        colDims[i] = cds[i/8];
    }

    int streamList[dims*8];
    for (int i=0; i<dims*8; i++) {
        streamList[i] = (i % 8) + 1;
    }

    int size = (sizeof(rowDims) / sizeof(rowDims[0]));
    int executionTimes[size*8];
    double flopRates[size*8];

    std::string algorithm;
    for (int numStreams = 1; numStreams <= 8; numStreams++) {
        printf("num streams: %d\n", numStreams);
        for (int j = 0; j < size; j++) {
            int M = rowDims[j];
            int N = colDims[j];
            printf("dims: %d, %d\n", M, N);
            int rowsPerBlock = (M + numStreams - 1) / numStreams;

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
            
            // allocate memory on device
            cudaMalloc( (void**) &mat_d, sizeof(double)*M*N);
            cudaMalloc( (void**) &vec_d, sizeof(double)*N);
            cudaMalloc( (void**) &res_d, sizeof(double)*M);

            //copy data from HOST to DEVICE
            cudaMemcpy(vec_d,vec_h,sizeof(double)*N,cudaMemcpyHostToDevice);
            cudaMemcpy(mat_d,mat_h,sizeof(double)*M*N,cudaMemcpyHostToDevice);

            struct timeval startTime;
            struct timeval endTime;

            gettimeofday(&startTime, nullptr);  
            gettimeofday(&endTime, nullptr);

            cudaStream_t streams[numStreams];

            for (int i=0; i<numStreams; i++) {
                cudaStreamCreate(&streams[i]);
            }

            for (int i=0; i<numStreams; i++) {
                cudaMemcpyAsync(&mat_d[i*rowsPerBlock*N], &mat_h[i*rowsPerBlock*N], sizeof(double)*rowsPerBlock*N, cudaMemcpyHostToDevice, streams[i]);

                if (e == NO_WARP) {
                    algorithm = "No Warp";

                    dim3 nthreads(256, 1, 1); // threads per block NOTE NOT MORE THAN 1024
                    dim3 nblocks ((rowsPerBlock + nthreads.x-1)/nthreads.x, 1, 1); // blocks per grid -> should be 1
                    gettimeofday(&startTime, nullptr);  
                    matVecMulNoWarp<<<nblocks, nthreads, 0, streams[i]>>>(M, N, &mat_d[i*rowsPerBlock*N], vec_d, &res_d[i*rowsPerBlock]);
                } else if (e == MULTI_WARP) {
                    algorithm = "Multiple warps per row";

                    dim3 nthreads(WARP_SIZE * nwarps,1,1);
                    dim3 nblocks(rowsPerBlock,1,1);
                    gettimeofday(&startTime, nullptr); 
                    matVecMultiWarp<<<nblocks, nthreads, 0, streams[i]>>>(M, N, &mat_d[i*rowsPerBlock*N], vec_d, &res_d[i*rowsPerBlock], i, rowsPerBlock);
                }

                cudaMemcpyAsync(&res_h[i*rowsPerBlock], &res_d[i*rowsPerBlock], sizeof(double)*rowsPerBlock, cudaMemcpyDeviceToHost, streams[i]);
            }

            cudaDeviceSynchronize();

            for (int i = 0; i < numStreams; i++) {
                cudaStreamDestroy(streams[i]);
            }

            gettimeofday(&endTime, nullptr);

            double flops = 2 * M * N;
            int microseconds = (endTime.tv_sec - startTime.tv_sec) * 1000000 + (endTime.tv_usec - startTime.tv_usec);
            double floprate = getFlopRate(flops, microseconds);

            executionTimes[j + (numStreams-1)*8] = microseconds;
            flopRates[j + (numStreams-1)*8] = floprate;

            std::cout << "M: " << M << ", N: " << N << ", Time: " << microseconds << ", Floprate: " << floprate << std::endl;

            //free memory 
            cudaFree(res_d);
            cudaFree(vec_d);
            cudaFree(mat_d);

            cudaFreeHost(mat_h);
            cudaFreeHost(vec_h);
            cudaFreeHost(res_h);
        }
    }

    generateLatexTable(rowDims, colDims, executionTimes, flopRates, streamList, size, algorithm, output);
}

void validate(Experiment e) {
    int M = 100;
    int N = 100;

    int numStreams = 4;

    int rowsPerBlock = (M + numStreams - 1) / numStreams;

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
    
    // allocate memory on device
    cudaMalloc( (void**) &mat_d, sizeof(double)*M*N);
    cudaMalloc( (void**) &vec_d, sizeof(double)*N);
    cudaMalloc( (void**) &res_d, sizeof(double)*M);

    //copy data from HOST to DEVICE
    cudaMemcpy(vec_d,vec_h,sizeof(double)*N,cudaMemcpyHostToDevice);
    cudaMemcpy(mat_d,mat_h,sizeof(double)*M*N,cudaMemcpyHostToDevice);

    struct timeval startTime;
    struct timeval endTime;

    gettimeofday(&startTime, nullptr);  
    gettimeofday(&endTime, nullptr);

    cudaStream_t streams[numStreams];

    for (int i=0; i<numStreams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    for (int i=0; i<numStreams; i++) {
        cudaMemcpyAsync(&mat_d[i*rowsPerBlock*N], &mat_h[i*rowsPerBlock*N], sizeof(double)*rowsPerBlock*N, cudaMemcpyHostToDevice, streams[i]);

        if (e == NO_WARP) {
            dim3 nthreads(256, 1, 1); // threads per block NOTE NOT MORE THAN 1024
            dim3 nblocks ((rowsPerBlock + nthreads.x-1)/nthreads.x, 1, 1); // blocks per grid -> should be 1
            gettimeofday(&startTime, nullptr);  
            matVecMulNoWarp<<<nblocks, nthreads, 0, streams[i]>>>(M, N, &mat_d[i*rowsPerBlock*N], vec_d, &res_d[i*rowsPerBlock]);
        } else if (e == MULTI_WARP) {
            dim3 nthreads(WARP_SIZE * nwarps,1,1);
            dim3 nblocks(rowsPerBlock,1,1);
            gettimeofday(&startTime, nullptr); 
            matVecMultiWarp<<<nblocks, nthreads, 0, streams[i]>>>(M, N, &mat_d[i*rowsPerBlock*N], vec_d, &res_d[i*rowsPerBlock], i, rowsPerBlock);
        }

        cudaMemcpyAsync(&res_h[i*rowsPerBlock], &res_d[i*rowsPerBlock], sizeof(double)*rowsPerBlock, cudaMemcpyDeviceToHost, streams[i]);
    }

    cudaDeviceSynchronize();

    for (int i = 0; i < numStreams; i++) {
        cudaStreamDestroy(streams[i]);
    }

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

    //free memory 
    cudaFree(res_d);
    cudaFree(vec_d);
    cudaFree(mat_d);

    cudaFreeHost(mat_h);
    cudaFreeHost(vec_h);
    cudaFreeHost(res_h);
}


int main(int argc, char *argv[]) {

    if ((argc < 2) || (argc > 4))
    {
        std::cerr << "Usage: " << argv[0] << " <algorithm>" << std::endl;
        return 1; // Return an error code indicating incorrect usage
    }

    std::string output;
    if (argc == 4) {
        std::string flag = std::string(argv[2]);
        if (flag == "-o") {
            output = std::string(argv[3]);
        }
    }

    Experiment e;
    std::string algorithm = std::string(argv[1]);
    if (algorithm == "NO_WARP") {
        e = NO_WARP;
    } else if (algorithm == "MULTI_WARP") {
        e = MULTI_WARP;
    } else {
        std::cerr << "Invalid algorithm" << std::endl;
    }

    int ncuda_devices = 0;
    cudaGetDeviceCount(&ncuda_devices);
    printf("ncuda_devices = %d\n",ncuda_devices);

    if (ncuda_devices == 0) {
        fprintf(stderr,"NO CUDA DEVICES EXITING\n");
        return 0;
    }
    cudaSetDevice(0);

    runExperiment(e, output);
    validate(e);
}