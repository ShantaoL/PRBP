// PRBP.c 
// CUDA version of Linear System Solver (Gaussian Elimination)
// Pivoting free
// REF: Accelerating Linear System Solutions Using Randomization Techiniques
// 	Baboulin, Dongarra, Herrmann and Tomov, 2013
// 
// AUTHOR: Shantao LI
// DATE: Dec/2014

#define FP float
#define MAXBIT 12

#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <math.h>
#include <curand.h>
#include <cuda_runtime.h>
#include "cusparse.h"
#include "cublas_v2.h"

__global__ void gpu_genbutterlies(double *rand, FP *temp, int n) {
  // The recursive butterfly matrix can be divided into 8 blocks
  // We calculate all the blocks separately
  // extern __shared__ FP val[];
  //FP *a = &rand[0], *b = &rand[n*2];
  // FP block[8];
  int worker = threadIdx.x + blockDim.x * blockIdx.x;
  
  // Map the random enties into the temp matrix
  for (int offset=0; offset <= 2*n; offset += 2*n) { 
    if (worker < n) {
      if (worker < n/4 || (worker < 3*n/4 && worker >= n/2)) { 
	temp[worker*2 + offset*2] = expf(rand[worker+offset]/10.);
	temp[worker*2+1 + offset*2] = expf(rand[worker+n/4+offset]/10.);
      }
       else {
	temp[worker*2+offset*2] = expf(rand[worker-n/4+offset]/10.0);
	temp[worker*2+1+offset*2] = -expf(rand[worker+offset]/10.0);
      }
      // Put the 0.5 scalar at the second matrix!
      if (worker < n/2) {
	temp[worker*2+2*n+offset*2] = expf(rand[worker+offset+n]/10.)/2.;
	temp[worker*2+2*n+1+offset*2] = expf(rand[worker+n*3/2+offset]/10.)/2.;
      }
      else {
	temp[worker*2+2*n+offset*2] = expf(rand[worker+n/2+offset]/10.)/2.;
	temp[worker*2+2*n+1+offset*2] = -expf(rand[worker+offset+n]/10.)/2.;
      } 
    } 
  }
}

// Apply recursive butterfly matrices to A and B to generate Ar, Br
__global__ void gpu_setdiag(FP *m, int n) {
  int worker = threadIdx.x + blockDim.x * blockIdx.x;
  if (worker < n) {
    m[(n+1)*worker] = 1.0f;
  }
}

void cpu_matrixmult(FP *a,FP *b, FP *c, int n) {
  FP cvalue;
  int index, indexa, indexb;
  for(int col=0;col < n; col++)
    for(int row=0;row < n; row++) {
      indexb = col;
      index = row * n + col;
      cvalue = 0.;
      for (indexa = row*n; indexa < (row*n + n); indexa++, indexb+=n) 
	cvalue += a[indexa]*b[indexb];
      c[index] -= cvalue; //NOTE: This calculates the diff between CPU and GPU computations.
    }
}


int main(int argc, char *argv[]) {

  int i, j; // loop counters

  int gpucount = 0; // Count of available GPUs

  int n, seed; // matrix dimension
  FP *a,*c;
  FP *dev_a, *dev_b, *dev_c, *dev_ub, 
     *dev_rresult, *dev_tmp, *dev_ar, *dev_v;
  double *dev_r;

  cudaEvent_t start, stop; // using cuda events to measure time
  float elapsed_time_ms; // which is applicable for asynchronous code also
  cudaError_t errorcode;

  cusparseOperation_t notrans = CUSPARSE_OPERATION_NON_TRANSPOSE,
		      trans = CUSPARSE_OPERATION_TRANSPOSE;

  // --------------------SET PARAMETERS AND DATA -----------------------

  errorcode = cudaGetDeviceCount(&gpucount);
  if (errorcode == cudaErrorNoDevice) {
    printf("No GPUs are visible\n");
    exit(-1);
  }
  else printf("Device count = %d\n",gpucount);

  if (argc < 3) {
    printf("Usage: prbp <matrix dim> <seed>\n"); 
    exit (-1);
  }

  n = atoi(argv[1]);
  seed = atoi(argv[2]);

  dim3 RandGrid(n/1024 + 1);
  dim3 RandBlock(1024);

  // dynamically allocated memory for arrays on host
  a = (FP*) malloc(n * n * sizeof(FP));
  c = (FP*) malloc(n * n * sizeof(FP));

  srand(12345);

  for(i=0; i < n; i++) {
    for(j=0; j < n; j++) {
      a[i * n + j] = (FP) rand() / (FP) RAND_MAX;
      // Generate the ground truth here
      c[i * n + j] = (FP) rand() / (FP) RAND_MAX;
    }
  }

  // ------------- COMPUTATION DONE ON GPU ----------------------------
  
  cudaMalloc((void**)&dev_a, n * n * sizeof(FP)); // allocate memory on device
  cudaMalloc((void**)&dev_b, n * n * sizeof(FP));
  cudaMalloc((void**)&dev_c, n * n * sizeof(FP));

  cudaMemcpy(dev_a, a , n * n * sizeof(FP) ,cudaMemcpyHostToDevice);
  cudaMemcpy(dev_c, c, n * n *sizeof(FP),cudaMemcpyHostToDevice);

  float alpha = 1.0f;
  float beta = 0.0f;

  cublasHandle_t blasHandle;
  cublasCreate(&blasHandle);
  cublasSgemm(blasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,\
      &alpha, dev_a, n, dev_c, n, &beta, dev_b, n);

  cudaEventCreate(&start); // instrument code to measure start time
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);
  // First, using crand to generate uniform random number
  curandGenerator_t gen;
  cudaMalloc((void**)&dev_r, 4*n*(sizeof(double)));
  cudaMalloc((void**)&dev_tmp, 8*n*sizeof(FP));
  cudaMalloc((void**)&dev_rresult, 8*n*(sizeof(FP)));
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, seed);
  curandGenerateUniformDouble(gen, dev_r, 4*n);

  cusparseHandle_t handle=0;
  cusparseMatDescr_t descr=0; 
  cusparseCreate(&handle);
  cusparseCreateMatDescr(&descr);

  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

  int *csrRowIndex, *csrColIndex, *dev_csrRowIndex, *dev_csrColIndex;
  csrRowIndex = (int*)calloc(4*(n+1), sizeof(int));
  csrColIndex = (int*)calloc(8*n, sizeof(int));

  // Set up the Row/Col indexes for first matrix in recursive butterfly
  for (int offset = 0; offset <= 2*n; offset+=2*n) {
    for (int i=0; i<n; i++) {
      csrRowIndex[i+offset+offset/n] = i*2;
      if (i < n/4 || (i >= n/2 && i < n*3/4)) {
	csrColIndex[2*i+2*offset] = i;
	csrColIndex[2*i+1+2*offset] = i + n/4; 
      }
      else {
	csrColIndex[2*i+2*offset] = i - n/4;
	csrColIndex[2*i+1+2*offset] = i;
      }
    }
    csrRowIndex[n+offset+offset/n] = 2*n;
    
    // The second matrix is easy to index
    for (int i=0; i<n; i++) {
      csrRowIndex[i+n+1+offset+offset/n] = i*2;
      if (i < n/2) {
	csrColIndex[2*i+2*n+2*offset] = i;
	csrColIndex[2*i+1+2*n+2*offset] = i + n/2;
      }
      else {
	csrColIndex[2*i+2*n+2*offset] = i - n/2;
	csrColIndex[2*i+1+2*n+2*offset] = i;
      }
    }
    csrRowIndex[2*n+offset+1+offset/n] = 2*n;
  }

  cudaMalloc((void**)&dev_csrRowIndex, 4*(n+1)*(sizeof(int)));
  cudaMalloc((void**)&dev_csrColIndex, 8*n*(sizeof(int)));
  cudaMemcpy(dev_csrRowIndex, csrRowIndex, 4*(n+1)*sizeof(int),
      cudaMemcpyHostToDevice);
  cudaMemcpy(dev_csrColIndex, csrColIndex, 8*n*sizeof(int), 
      cudaMemcpyHostToDevice);

  gpu_genbutterlies<<<RandGrid, RandBlock>>>(dev_r, dev_tmp, n); 
  cudaDeviceSynchronize();
 
  int *dev_resultColIndex, *dev_resultRowIndex;

  cudaMalloc((void**)&dev_resultRowIndex, 2*(n+1)*(sizeof(int)));
  cudaMalloc((void**)&dev_resultColIndex, 8*n*(sizeof(int)));

  // Step 1
  int nnz = 0;
  int *dev_nnz = &nnz;
  cusparseXcsrgemmNnz(handle, notrans, notrans, n, n, n, descr, 2*n, 
      dev_csrRowIndex, dev_csrColIndex, descr, 2*n, &dev_csrRowIndex[n+1],
      &dev_csrColIndex[2*n], descr, dev_resultRowIndex, dev_nnz);
  cusparseXcsrgemmNnz(handle, notrans, notrans, n, n, n, descr, 2*n,
     &dev_csrRowIndex[2*(n+1)], &dev_csrColIndex[4*n], descr, 2*n, 
     &dev_csrRowIndex[3*(n+1)], &dev_csrColIndex[6*n], descr, 
     &dev_resultRowIndex[n+1], dev_nnz);
  // Step 2
  cusparseScsrgemm(handle, notrans, notrans, n, n, n, descr, 2*n, dev_tmp, 
    dev_csrRowIndex, dev_csrColIndex, descr, 2*n, &dev_tmp[2*n], &dev_csrRowIndex[n+1], 
    &dev_csrColIndex[2*n], descr, dev_rresult, dev_resultRowIndex, dev_resultColIndex);
  cusparseScsrgemm(handle, notrans, notrans, n, n, n, descr, 2*n, 
      &dev_tmp[4*n], &dev_csrRowIndex[2*n+2], &dev_csrColIndex[4*n], descr, 2*n, 
      &dev_tmp[6*n], &dev_csrRowIndex[3*n+3], &dev_csrColIndex[6*n], descr,
      &dev_rresult[4*n], &dev_resultRowIndex[n+1], &dev_resultColIndex[4*n]);

  // Load in A & B, want to calcualte Ar = U^tAV and U^tB
  cudaMalloc((void**)&dev_ub, n * n * sizeof(FP));
  cudaMalloc((void**)&dev_ar, n * n * sizeof(FP));
  
  // Now we want to do Ar and U^tB
  // Notice both A and B are dense matrices
  cusparseScsrmm(handle, trans, n, n, n, 4*n, &alpha, descr, dev_rresult,
      dev_resultRowIndex, dev_resultColIndex, dev_a, n, &beta, dev_c, n);
  cusparseScsrmm(handle, trans, n, n, n, 4*n, &alpha, descr, dev_rresult,
      dev_resultRowIndex, dev_resultColIndex, dev_b, n, &beta, dev_ub, n);
  // Convert V to dense format and use BLAS3 to calculate dev_ar
  cudaMalloc((void**)&dev_v, n*n*sizeof(FP));
  cusparseScsr2dense(handle, n, n, descr, &dev_rresult[4*n], 
      &dev_resultRowIndex[n+1], &dev_resultColIndex[4*n], dev_v, n);
  cublasSgemm(blasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,\
      &alpha, dev_c, n, dev_v, n, &beta, dev_ar, n);
  
  // First do LU decomposition get triangle matrices L & U
  // Then solve LZ = UB & UY = Z
  // So X = VY
  int info, *dev_info;// *pivotArray, *dev_pivot;
  cudaMalloc((void**)&dev_info, sizeof(int));
  float **dev_array, *array[] = { dev_ar };
  cudaMalloc((void**)&dev_array, sizeof(array));
  cudaMemcpy(dev_array, array, sizeof(array), cudaMemcpyHostToDevice);
  cublasSgetrfBatched(blasHandle, n, dev_array, n, NULL, dev_info, 1);
  cudaMemcpy(&info, dev_info, sizeof(int), cudaMemcpyDeviceToHost);
  if (info != 0) {
    printf("LU decomp. failed!: %d\n", info);
    exit(1);
  }

  cudaMemcpy(dev_b, dev_ar, n*n*sizeof(FP), cudaMemcpyDeviceToDevice);
  gpu_setdiag<<<RandGrid, RandBlock>>>(dev_b, n);
  
  cublasStrsm(blasHandle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, \
      CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, n, &alpha, dev_b, n, dev_ub, n);

  cublasStrsm(blasHandle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, \
      CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, n, &alpha, dev_ar, n, dev_ub, n);

  cublasSgemm(blasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,\
            &alpha, dev_v, n, dev_ub, n, &beta, dev_c, n);

  cudaEventRecord(stop, 0); // instrument code to measure end time
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed_time_ms, start, stop);
  printf("Time to calculate results on GPU: %f ms.\n", elapsed_time_ms); 
  
  FP *result = (FP*)malloc(n*n*sizeof(FP));
  cudaMemcpy(result,dev_c,n*n*sizeof(FP) ,cudaMemcpyDeviceToHost);
  FP err = 0;
  for (int i=0; i<n*n; i++) {
    err += abs(c[i] - result[i]);
  }
  printf("L1-norm of error is: %e\n", err/(n*n));
  
// -------------- clean up ---------------------------------------
  free(a);
  free(c);
  free(csrRowIndex);
  free(csrColIndex);
  free(result);
  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_ub);
  cudaFree(dev_ar);
  cudaFree(dev_c);
  cudaFree(dev_r);
  cudaFree(dev_tmp);
  cudaFree(dev_v);
  cudaFree(dev_rresult);
  cudaFree(dev_csrRowIndex);
  cudaFree(dev_csrColIndex);
  cudaFree(dev_resultRowIndex);
  cudaFree(dev_resultColIndex);
  cudaFree(dev_info);
  cudaFree(dev_array);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  curandDestroyGenerator(gen);
  cusparseDestroyMatDescr(descr);
  cusparseDestroy(handle);
  cublasDestroy(blasHandle);
  
  return 0;
}
