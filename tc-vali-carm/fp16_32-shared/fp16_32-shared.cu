#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>

#include "../../nvml_tools.cu"

#define M 16
#define N 8
#define K 16

#define THREADS_PER_BLOCK 1024
#define NUM_BLOCKS 32768
#define A_SIZE M *K *(THREADS_PER_BLOCK / 32) * NUM_BLOCKS
#define B_SIZE K *N *(THREADS_PER_BLOCK / 32) * NUM_BLOCKS
#define C_SIZE M *N *(THREADS_PER_BLOCK / 32) * NUM_BLOCKS
#define ITERATIONS 32768 / 4

#define MEM 6
#define FLOP 2

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
  cudaSetDevice(0);
}

// Kernel function
__global__ void benchmark_alt(float *d_A, float *d_B, float *d_C,
                              uint64_t *d_startClk, uint64_t *d_stopClk,
                              uint64_t *d_timeStart, uint64_t *d_timeStop) {
  // Code to be executed on the GPU
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t start = 0;
  uint64_t stop = 0;
  uint64_t time_start = 0;
  uint64_t time_stop = 0;

  __shared__ float s[THREADS_PER_BLOCK];

  // create registers for threads
  half fragsA[8];
  half fragsB[4];
  float fragsC[4];

  for (int i = 0; i < 8; i++) {
    fragsA[i] = d_A[i + id * 8];
  }
  for (int i = 0; i < 4; i++) {
    fragsB[i] = d_B[i + id * 4];
    fragsC[i] = d_C[i + id * 4];
  }

  uint32_t const *A = reinterpret_cast<uint32_t const *>(
      &fragsA[0]);  // change from half to bit 32 which is what the mma takes
  uint32_t const *B = reinterpret_cast<uint32_t const *>(&fragsB[0]);
  float *C = reinterpret_cast<float *>(&fragsC[0]);

  // start timing
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(time_start)::"memory");
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");

  for (int i = 0; i < ITERATIONS; i++) {
#pragma unroll
    for (int j = 0; j < MEM; j++) {
      fragsC[0] = s[threadIdx.x];
      s[THREADS_PER_BLOCK - threadIdx.x - 1] = fragsC[0];
    }
#pragma unroll
    for (int j = 0; j < FLOP; j++) {
      // assembly mma
      asm volatile(
          "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]));
    }
  }
  // stop timing
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop)::"memory");
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(time_stop)::"memory");

  for (int i = 0; i < 4; i++) {
    d_C[i + id * 4] = fragsC[i];
  }

  d_startClk[id] = start;
  d_stopClk[id] = stop;
  d_timeStart[id] = time_start;
  d_timeStop[id] = time_stop;
}

// D = A*B + D
int main() {
  // Code to be executed on the CPU
  // start nvml
  // thread to measure power configuration
  std::thread measuring_thread;
  monitor_args thread_args;
  thread_args.powerArray = std::vector<int>();
  thread_args.clockArray = std::vector<int>();
  thread_args.flag = 0;

  init_nvml(&thread_args, &measuring_thread, false);
  cudaCheckError(cudaDeviceSynchronize());

  // Print CUDA info
  printCudaInfo();

  // Calculate matrix dimensions
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
  uint64_t *startClk =
      (uint64_t *)malloc(NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(uint64_t));
  uint64_t *stopClk =
      (uint64_t *)malloc(NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(uint64_t));

  uint64_t *d_startClk, *d_stopClk;
  cudaCheckError(cudaMalloc((void **)&d_startClk,
                            NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(uint64_t)));
  cudaCheckError(cudaMalloc((void **)&d_stopClk,
                            NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(uint64_t)));

  // handle timings
  uint64_t *timeStart =
      (uint64_t *)malloc(NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(uint64_t));
  uint64_t *stopStop =
      (uint64_t *)malloc(NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(uint64_t));

  uint64_t *d_timeStart, *d_timeStop;
  cudaCheckError(cudaMalloc((void **)&d_timeStart,
                            NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(uint64_t)));
  cudaCheckError(cudaMalloc((void **)&d_timeStop,
                            NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(uint64_t)));

  thread_args.flag = 1;
  // Launch kernel on the GPU
  benchmark_alt<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
      d_A, d_B, d_C, d_startClk, d_stopClk, d_timeStart, d_timeStop);

  // Wait for GPU to finish
  cudaCheckError(cudaDeviceSynchronize());
  thread_args.flag = 0;
  stop_nvml(&measuring_thread, thread_args.powerArray, thread_args.clockArray);

  // Copy device memory to host
  cudaCheckError(cudaMemcpy(startClk, d_startClk,
                            NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(uint64_t),
                            cudaMemcpyDeviceToHost));
  cudaCheckError(cudaMemcpy(stopClk, d_stopClk,
                            NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(uint64_t),
                            cudaMemcpyDeviceToHost));
  cudaCheckError(cudaMemcpy(timeStart, d_timeStart,
                            NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(uint64_t),
                            cudaMemcpyDeviceToHost));
  cudaCheckError(cudaMemcpy(stopStop, d_timeStop,
                            NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(uint64_t),
                            cudaMemcpyDeviceToHost));

  cudaCheckError(cudaDeviceSynchronize());

  uint64_t total_clk =
      *std::max_element(stopClk, stopClk + NUM_BLOCKS * THREADS_PER_BLOCK) -
      *std::min_element(startClk, startClk + NUM_BLOCKS * THREADS_PER_BLOCK);
  double total_time =
      *std::max_element(stopStop, stopStop + NUM_BLOCKS * THREADS_PER_BLOCK) -
      *std::min_element(timeStart, timeStart + NUM_BLOCKS * THREADS_PER_BLOCK);

  total_time = total_time / 1e9;

  uint64_t fma = (uint64_t)M * N * K * ITERATIONS * (THREADS_PER_BLOCK / 32) *
                 NUM_BLOCKS * FLOP;

  long bytes = sizeof(float) * 2 * ITERATIONS * THREADS_PER_BLOCK * NUM_BLOCKS *
               MEM;  // 2 for read and write

  double bw = (float)bytes / (float)total_time / 1e9;

  double FLOPS = fma * 2 / total_time / 1e12;

  std::cout << "Bandwidth = " << bw << " (FMA/clk/SM)\n";

  std::cout << "FLOPS = " << FLOPS << "(TFLOPs) \n";
  std::cout << "AI = " << fma * 2. / bytes << " (FLOP/byte)\n";

  std::cout << "Total Clk number = " << total_clk << "\n";

  std::cout << "Total Time number = " << total_time << " (sec)\n";
  std::cout << "Average Clock Frequency = " << total_clk / total_time / 1e6
            << " (MHz)\n";

  std::cout << "---------------------------------------------------------\n";

  // Free device memory
  cudaCheckError(cudaFree(d_A));
  cudaCheckError(cudaFree(d_B));
  cudaCheckError(cudaFree(d_C));
  cudaCheckError(cudaFree(d_startClk));
  cudaCheckError(cudaFree(d_stopClk));
  cudaCheckError(cudaFree(d_timeStart));
  cudaCheckError(cudaFree(d_timeStop));

  // Free host memory
  free(h_A);
  free(h_B);
  free(h_C);
  free(startClk);
  free(stopClk);
  free(timeStart);
  free(stopStop);

  return 0;
}
