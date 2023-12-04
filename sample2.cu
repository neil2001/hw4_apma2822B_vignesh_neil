#include <iostream>
#include <cstdlib>
#include <ctime>
#include <sys/time.h>
#include <fstream>
#include <vector>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <string>

#define FULL_MASK 0xffffffff
#define WARP_SIZE 32


__global__
void testKernel(double* matrix, double* vector, 
							int numRowsM, int vecLength, int streamIndex, int streamSize,
							double* result) {
	printf("Stream index: %d\n", streamIndex);
}

__global__
void mvKernelNoWarp(double* matrix, double* vector, 
							int numRowsM, int vecLength, int streamIndex, int streamSize,
							double* result) {
	int row = streamIndex * streamSize + blockIdx.x;
	if (row < numRowsM) {
		for (int col = 0; col < vecLength; col ++) {
			if (col < vecLength) {
				result[blockIdx.x] += matrix[col+blockIdx.x*vecLength] * vector[col];
			}
		}
	}
}

__global__
void mvKernelMultipleWarps(double* matrix, double* vector, 
							int numRowsM, int vecLength, int streamIndex, int streamSize,
							double* result) {
	
	int row = streamIndex * streamSize + blockIdx.x;
	// printf("Row: %d\n", row);
	int lane = threadIdx.x % WARP_SIZE;
	int warpid = threadIdx.x/WARP_SIZE;
	int nwarps = blockDim.x/WARP_SIZE;

	double sum = 0.0;

	if (row < numRowsM) {
		for (int col = lane + WARP_SIZE*warpid; col < vecLength; col += WARP_SIZE*nwarps) { // modulus addition
			if (col < vecLength) {
				// sum += matrix[row*vecLength + col]  * vector[col];
				sum += matrix[col+blockIdx.x*vecLength]  * vector[col];
			}
		}
		__syncwarp();
		for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
			sum += __shfl_down_sync(FULL_MASK, sum, offset);
		}

		__shared__ double s_mem[1024/WARP_SIZE]; // max 32 warps per block 
		if (lane == 0) {
			s_mem[warpid] = sum;
		}

		__syncthreads(); // sync threads within block
		if (threadIdx.x == 0) { // first lane in first warp
			for (int j = 0; j < nwarps; ++j) {
				// result[row] += s_mem[j];
				result[blockIdx.x] += s_mem[j];

			}   
		}
	}
}

void matVecMul(double* mat_h, double* vec_h, double* result_h, int M, int numRowsM, int vecLength) {
		double *mat_d, *vec_d, *result_d;

		cudaMalloc(&mat_d, numRowsM * vecLength * sizeof(double));
		cudaMalloc(&vec_d, vecLength * sizeof(double));
		cudaMalloc(&result_d, numRowsM * sizeof(double));

		int nwarps = 32; // max 32 nwarps per block
		int streamNumRows = (numRowsM + M - 1)/M; // ceil

		cudaMemcpy(vec_d, vec_h, vecLength * sizeof(double), cudaMemcpyHostToDevice); // all 

		cudaStream_t streams[M];

		struct timeval start;
		struct timeval end;


		dim3 nthreads(WARP_SIZE * nwarps,1,1);
		dim3 nblocks(streamNumRows,1,1);

		// dim3 nthreads(1,1,1);
		// dim3 nblocks(streamNumRows,1,1);
		// Copy data and launch kernels in each stream
		gettimeofday(&start, 0);
		for (int i = 0; i < M; ++i) {
			cudaStreamCreate(&streams[i]); // time this asw
		}
		gettimeofday(&end, 0);
		printf("Time to make %d streams: %ld\n", M, (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec));


		for (int i = 0; i < M; ++i) {
				int offset = i * streamNumRows; // row to start stream at
				cudaMemcpyAsync(&mat_d[offset*vecLength], &mat_h[offset*vecLength], streamNumRows * vecLength * sizeof(double), cudaMemcpyHostToDevice, streams[i]);
				// testKernel<<<nblocks, nthreads, 0, streams[i]>>>(&mat_d[offset*vecLength], vec_d, numRowsM, vecLength, i, streamNumRows, &result_d[offset]);
				// mvKernelNoWarp<<<nblocks, nthreads, 0, streams[i]>>>(&mat_d[offset*vecLength], vec_d, numRowsM, vecLength, i, streamNumRows, &result_d[offset]);
				mvKernelMultipleWarps<<<nblocks, nthreads, 0, streams[i]>>>(&mat_d[offset*vecLength], vec_d, numRowsM, vecLength, i, streamNumRows, &result_d[offset]);
				cudaMemcpyAsync(&result_h[offset], &result_d[offset], streamNumRows * sizeof(double), cudaMemcpyDeviceToHost, streams[i]);
		}
		
		cudaDeviceSynchronize();
		fflush(stdout);

		gettimeofday(&start, 0);
		for (int i = 0; i < M; ++i) {
			cudaStreamDestroy(streams[i]);
		}
		cudaMemcpy(result_h, result_d, numRowsM * sizeof(double), cudaMemcpyDeviceToHost);

		gettimeofday(&end, 0);
		printf("Time to destroy %d streams: %lu\n", M, (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec));

		cudaFree(mat_d);
		cudaFree(vec_d);
		cudaFree(result_d);

}

void deleteMatrix(double** matrix, int numRows) {
  for (int i = 0; i < numRows; ++i) {
    delete[] matrix[i];
  }
  delete[] matrix;
}


double* generateRandomMatrixContiguous(int numRows, int numCols) {
	double* matrix;
	cudaHostAlloc((void**)&matrix, numRows * numCols * sizeof(double), cudaHostAllocWriteCombined);

	for (int i = 0; i < numRows * numCols; ++i) {
		matrix[i] = 2;
		//static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
	}
	return matrix;
}

double* generateRandomVector(int length) {
	double* vector;
	cudaHostAlloc((void**)&vector, length * sizeof(double), cudaHostAllocWriteCombined);

	for (int i = 0; i < length; ++i) {
		vector[i] = 2;//static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
	}
	return vector;
}


unsigned long timedMVMult(int numRows, int numCols, int M) {

	srand(0);

	double* matrix = generateRandomMatrixContiguous(numRows, numCols);
	double* vector = generateRandomVector(numCols);

	double* result_h;
	cudaHostAlloc((void**)&result_h, numRows * sizeof(double), cudaHostAllocDefault);

	struct timeval start;
	struct timeval end;

	int numRowsM = numRows;
	int vecLength = numCols;

	gettimeofday(&start, 0);
	
	matVecMul(matrix, vector, result_h, M, numRowsM, vecLength);
	for (int i = 0; i < numRowsM; i++) {
				std::cout << result_h[i] << " ";
		}
	std::cout << std::endl;
	printf("NumRowsM: %d\n", numRowsM);
	printf("VecLength: %d\n", vecLength);
	printf("M: %d\n", M);
	// saveVector("outputs/result.txt", result_h, numRowsM);
	gettimeofday(&end, 0);

	cudaFreeHost(matrix);
	cudaFreeHost(vector);
	cudaFreeHost(result_h);
	
	return (end.tv_sec - start.tv_sec) * 1000000 + (end.tv_usec - start.tv_usec);
}

double meanOfMatrix(double** matrix, int numRows, int numCols) {
	double sum = 0.0;
	int count = 0;
	for (int r = 0; r < numRows; ++r) {
		for (int c = 0; c < numCols; ++c) {
			if (std::isfinite(matrix[r][c])) {
					sum += matrix[r][c];
					count++;
			}
		}
	}
	std::cout << "Mean of matrix is: " << sum/(numRows*numCols) << std::endl;
	return sum/(numRows*numCols);
}

void saveLatex(const char* fileName, double** times, double** floprate, int numRows, bool inclTime=true) {
		std::ofstream outputFile(fileName);

		if (!outputFile.is_open()) {
				std::cerr << "Failed to open the file for writing." << std::endl;
				return;
		}

		for (int r = 0; r < numRows; ++r) {

			outputFile << "\\hline" << std::endl;
			outputFile << 1000 + 500*r << " & " << 1000 + 500*r;
			for (int M = 1; M <= 8; ++M) {

				if (inclTime) {
					outputFile <<  " & " << M << \
											" & " << times[r][M-1] << " & " << floprate[r][M-1] << " \\\\" << std::endl;
				} else {
					outputFile <<  " & " << M << \
											" & " << floprate[r][M-1] << " \\\\" << std::endl;
				}

			}
		}

		outputFile << "\\hline" << std::endl;
		outputFile << "Mean floprate " << meanOfMatrix(floprate, numRows, 8) << std::endl;
		outputFile.close();
		std::cout << "Matrix has been saved to " << fileName << std::endl;
}


int main(int argc, char *argv[]) {
	double** times = new double*[3];
	double** floprate = new double*[3];

	int rowCount;
	int colCount;
	
	// rowCount = 1000;
	// colCount = rowCount;
	// unsigned long elapsed_time = timedMVMult(rowCount, colCount, 8);
	for (int numRows = 0; numRows < 3; numRows +=1) {

	  rowCount = 1000 + 500*numRows;
	  colCount = rowCount;

	  times[numRows] = new double[8];
	  floprate[numRows] = new double[8];

	  for (int M = 1; M <= 8; M += 1) {
	    unsigned long elapsed_time = timedMVMult(rowCount, colCount, M);
	    if (elapsed_time == 0) {
	      elapsed_time = 1;
	    }
	    times[numRows][M] = elapsed_time;
	    floprate[numRows][M] = (2 * rowCount * colCount)/elapsed_time * std::pow(10, -6);
	  }

	}

	meanOfMatrix(floprate, 3, 8);
	saveLatex("outputs/time_floprate.txt", times, floprate, 3, true);

	deleteMatrix(times, 3);
	deleteMatrix(floprate, 3);

	return 0;
}