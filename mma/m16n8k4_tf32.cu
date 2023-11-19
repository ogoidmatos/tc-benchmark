#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>

#define THREADS_PER_BLOCK 256
#define A_SIZE 16 * 4 * (THREADS_PER_BLOCK / 32)
#define B_SIZE 8 * 4 * (THREADS_PER_BLOCK / 32)
#define C_SIZE 16 * 8 * (THREADS_PER_BLOCK / 32)
#define ITERATIONS 1024

#define DEBUG
#ifdef DEBUG
#define cudaCheckError(ans) \
  { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line,
                       bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}
#else
#define cudaCheckError(ans) ans
#endif

void printCudaInfo() {
  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);

  printf("---------------------------------------------------------\n");
  printf("Found %d CUDA devices\n", deviceCount);

  for (int i = 0; i < deviceCount; i++) {
    cudaDeviceProp deviceProps;
    cudaGetDeviceProperties(&deviceProps, i);
    printf("Device %d: %s\n", i, deviceProps.name);
    printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
    printf("   Global mem: %.0f MB\n",
           static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
    printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
  }
  printf("---------------------------------------------------------\n");
}

// Kernel function
__global__ void benchmark(float *d_A, float *d_B, float *d_C,
                          uint64_t *d_startClk, uint64_t *d_stopClk) {
  // Code to be executed on the GPU
  int id = blockIdx.x * blockDim.x threadIdx.x;
  uint64_t start = 0;
  uint64_t stop = 0;
  // declare shared memory
  __shared__ float shared_A[A_SIZE];
  __shared__ float shared_B[B_SIZE];
  __shared__ float shared_C[C_SIZE];

  // initialize shared memory
  for (int i = 0; i < A_SIZE; i++) {
    shared_A[i] = d_A[i];
  }
  for (int i = 0; i < B_SIZE; i++) {
    shared_B[i] = d_B[i];
  }
  for (int i = 0; i < C_SIZE; i++) {
    shared_C[i] = d_C[i];
  }

  // synchronize threads
  asm volatile("bar.sync 0;");

  // assembly ldmatrix
  asm volatile(
      ".reg .b32 %%a<2>;\n\t"
      "ldmatrix.sync.aligned.m8n8.x2.b16 {%%a0, %%a1}, [%0];\n\t" ::"r"(
          (unsigned)shared_A[id]));
  asm volatile(
      ".reg .b32 b0;\n\t"
      "ldmatrix.sync.aligned.m8n8.x1.trans.b16 {b0}, [%0];\n\t" ::"r"(
          (unsigned)shared_B[id]));
  asm volatile(
      ".reg .b32 %%c<4>;\n\t"
      "ldmatrix.sync.aligned.m8n8.x4.b16 {%%c0,%%c1,%%c2,%%c3}, [%0];\n\t" ::
          "r"((unsigned)shared_C[id]));

  // synchronize threads
  asm volatile("bar.sync 0;");

  // start timing
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");
  for (int i = 0; i < ITERATIONS; i++) {
    // assembly mma
    asm volatile(
        "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32 "
        "{%%c0,%%c1,%%c2,%%c3}, {%%a0,%%a1}, {b0}, {%%c0,%%c1,%%c2,%%c3};\n\t");
    __syncwarp();
  }
  // stop timing
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop)::"memory");

  d_startClk[id] = start;
  d_stopClk[id] = stop;
}

// D = A*B + D
int main() {
  // Code to be executed on the CPU

  // Print CUDA info
  printCudaInfo();

  // Calculate matrix dimensions
  int BLOCKS = 1;
  int dimA = A_SIZE;
  int dimB = B_SIZE;
  int dimC = C_SIZE;  // dimC is the same as dimD

  // Allocate host memory
  float *h_A = (float *)malloc(dimA * sizeof(float));
  float *h_B = (float *)malloc(dimB * sizeof(float));
  float *h_C = (float *)malloc(dimC * sizeof(float));

  // Initialize host memory
  for (int i = 0; i < dimA; i++) {
    h_A[i] = 0.0f;
  }
  for (int i = 0; i < dimB; i++) {
    h_B[i] = 0.0f;
  }
  for (int i = 0; i < dimC; i++) {
    h_C[i] = 0.0f;
  }

  // Allocate device memory
  float *d_A, *d_B, *d_C;
  cudaCheckError(cudaMalloc((void **)&d_A, dimA * sizeof(float)));
  cudaCheckError(cudaMalloc((void **)&d_B, dimB * sizeof(float)));
  cudaCheckError(cudaMalloc((void **)&d_C, dimC * sizeof(float)));

  // Copy host memory to device
  cudaCheckError(
      cudaMemcpy(d_A, h_A, dimA * sizeof(float), cudaMemcpyHostToDevice));
  cudaCheckError(
      cudaMemcpy(d_B, h_B, dimB * sizeof(float), cudaMemcpyHostToDevice));
  cudaCheckError(
      cudaMemcpy(d_C, h_C, dimC * sizeof(float), cudaMemcpyHostToDevice));

  // handle clock
  uint64_t *startClk = (uint64_t *)malloc(THREADS_PER_BLOCK * sizeof(uint64_t));
  uint64_t *stopClk = (uint64_t *)malloc(THREADS_PER_BLOCK * sizeof(uint64_t));

  uint64_t *d_startClk, *d_stopClk;
  cudaCheckError(
      cudaMalloc((void **)&d_startClk, THREADS_PER_BLOCK * sizeof(uint64_t)));
  cudaCheckError(
      cudaMalloc((void **)&d_stopClk, THREADS_PER_BLOCK * sizeof(uint64_t)));

  // Launch kernel on the GPU
  benchmark<<<BLOCKS, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, d_startClk,
                                           d_stopClk);

  // Wait for GPU to finish
  cudaCheckError(cudaDeviceSynchronize());

  // Copy device memory to host
  cudaCheckError(cudaMemcpy(startClk, d_startClk,
                            THREADS_PER_BLOCK * sizeof(uint64_t),
                            cudaMemcpyDeviceToHost));
  cudaCheckError(cudaMemcpy(stopClk, d_stopClk,
                            THREADS_PER_BLOCK * sizeof(uint64_t),
                            cudaMemcpyDeviceToHost));

  uint64_t total_time =
      *std::max_element(stopClk, stopClk + THREADS_PER_BLOCK) -
      *std::min_element(startClk, startClk + THREADS_PER_BLOCK);

  uint64_t fma = 4 * 8 * 16 * ITERATIONS * THREADS_PER_BLOCK / 32;
  float bw = (float)fma / (float)total_time;

  std::cout << "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32  latency "
            << (float)total_time / (float)ITERATIONS << " cycles\n";
  std::cout << "FMA tensor bandwidth = " << bw << "(FMA/clk/SM)\n";

  std::cout << "Total Clk number = " << total_time << "\n";

  // Free device memory
  cudaCheckError(cudaFree(d_A));
  cudaCheckError(cudaFree(d_B));
  cudaCheckError(cudaFree(d_C));
  cudaCheckError(cudaFree(d_startClk));
  cudaCheckError(cudaFree(d_stopClk));

  // Free host memory
  free(h_A);
  free(h_B);
  free(h_C);
  free(startClk);
  free(stopClk);

  return 0;
}
