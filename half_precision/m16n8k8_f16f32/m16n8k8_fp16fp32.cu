#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>

#define M 16
#define N 8
#define K 8

#define THREADS_PER_BLOCK 1024
#define A_SIZE M *K *(THREADS_PER_BLOCK / 32)
#define B_SIZE K *N *(THREADS_PER_BLOCK / 32)
#define C_SIZE M *N *(THREADS_PER_BLOCK / 32)
#define ITERATIONS 32768

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
__global__ void benchmark_alt(float *d_A, float *d_B, float *d_C,
                              uint64_t *d_startClk, uint64_t *d_stopClk) {
  // Code to be executed on the GPU
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t start = 0;
  uint64_t stop = 0;

  // create registers for threads
  half fragsA[4];
  half fragsB[2];
  float fragsC[4];

  for (int i = 0; i < 4; i++) {
    fragsA[i] = d_A[i + id * 4];
    fragsC[i] = d_C[i + id * 4];
  }
  for (int i = 0; i < 2; i++) {
    fragsB[i] = d_B[i + id * 2];
  }

  uint32_t const *A = reinterpret_cast<uint32_t const *>(
      &fragsA[0]);  // change from half to bit 32 which is what the mma takes
  uint32_t const *B = reinterpret_cast<uint32_t const *>(&fragsB[0]);
  float *C = reinterpret_cast<float *>(&fragsC[0]);

  // synchronize threads
  asm volatile("bar.sync 0;");

  // start timing
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");

  for (int i = 0; i < ITERATIONS; i++) {
    // assembly mma
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(B[0]));
    //__syncwarp();
  }
  // stop timing
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop)::"memory");

  for (int i = 0; i < 4; i++) {
    d_C[i + id * 4] = fragsC[i];
  }

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
  benchmark_alt<<<BLOCKS, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, d_startClk,
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

  cudaCheckError(cudaDeviceSynchronize());

  uint64_t total_time =
      *std::max_element(stopClk, stopClk + THREADS_PER_BLOCK) -
      *std::min_element(startClk, startClk + THREADS_PER_BLOCK);

  uint64_t fma = (uint64_t)M * N * K * ITERATIONS * (THREADS_PER_BLOCK / 32);
  float bw = (float)fma / (float)total_time;

  std::cout << "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32  latency "
            << (float)total_time / (float)ITERATIONS << " cycles\n";
  std::cout << "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32  FMA Count "
            << fma << "\n";
  std::cout << "FMA tensor bandwidth = " << bw << " (FMA/clk/SM)\n";

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
