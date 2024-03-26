#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>

#include <cstdlib>
#include <iostream>
#include <vector>

#include "../../nvml_tools.cu"

#define THREADS_PER_BLOCK 1024
#define NUM_BLOCKS 32768L
#define ITERATIONS 32768L
#define MEM 2
#define FLOP 256
#define AI ((float)FLOP / MEM)

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
    printf("   Clock:      %.2f MHz\n", (deviceProps.clockRate / 1000.0f));
  }
  printf("---------------------------------------------------------\n");
}

// Kernel function
template <class T>
__global__ void benchmark_alt(T *d_X, T *d_Y, uint64_t *d_startClk,
                              uint64_t *d_stopClk, uint64_t *d_timeStart,
                              uint64_t *d_timeStop) {
  // Code to be executed on the GPU
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t start = 0;
  uint64_t stop = 0;
  uint64_t time_start = 0;
  uint64_t time_stop = 0;

  T a = (T)id;
  T b = a + 1;
  T c = b + 1;
  T d = c + 1;
  // synchronize threads
  // asm volatile("bar.sync 0;");

  // start timing
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(time_start)::"memory");
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");

  // #pragma unroll
  for (int i = 0; i < ITERATIONS; i++) {
//  assembly mma
#pragma unroll
    for (int j = 0; j < MEM; j++) {
      a = d_X[id];
      d_Y[id] = a;
    }
#pragma unroll
    for (int j = 0; j < FLOP; j++) {
      a = a * a + b;
      b = b * b + c;
      c = c * c + d;
      d = d * d + a;
    }
  }

  // // stop timing
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop)::"memory");
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(time_stop)::"memory");

  d_startClk[id] = start;
  d_stopClk[id] = stop;
  d_timeStart[id] = time_start;
  d_timeStop[id] = time_stop;
  d_X[id] = d;
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

  init_nvml(&thread_args, &measuring_thread);
  cudaCheckError(cudaDeviceSynchronize());

  // Print CUDA info
  printCudaInfo();

  float *h_X = (float *)malloc(NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(float));
  float *h_Y = (float *)malloc(NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(float));
  float *d_X, *d_Y;
  cudaCheckError(cudaMalloc((void **)&d_X,
                            NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(float)));
  cudaCheckError(cudaMemcpy(d_X, h_X,
                            NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(float),
                            cudaMemcpyHostToDevice));
  cudaCheckError(cudaMalloc((void **)&d_Y,
                            NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(float)));
  cudaCheckError(cudaMemcpy(d_Y, h_Y,
                            NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(float),
                            cudaMemcpyHostToDevice));

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
  benchmark_alt<float><<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
      d_X, d_Y, d_startClk, d_stopClk, d_timeStart, d_timeStop);

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

  long fma = 4 * ITERATIONS * THREADS_PER_BLOCK * NUM_BLOCKS *
             FLOP;  // 4 fma instructions, 4*2 flops

  long bytes = sizeof(float) * 2 * ITERATIONS * THREADS_PER_BLOCK * NUM_BLOCKS *
               MEM;  // 2 for read and write

  // float fma_bw = (float)fma / (float)total_clk;

  double FLOPS = fma * 2 / total_time / 1e12;

  double bw = (float)bytes / (float)total_time / 1e9;

  // std::cout << "FMA tensor bandwidth = " << bw << " (FMA/clk/SM)\n";
  std::cout << "Bandwidth = " << bw << " (GB/s)\n";
  std::cout << "FLOPS = " << FLOPS << "(TFLOPs) \n";
  std::cout << "AI = " << AI << " (FLOP/byte)\n";

  std::cout << "Total Clk number = " << total_clk << "\n";

  std::cout << "Total Time number = " << total_time << " (sec)\n";
  std::cout << "Average Clock Frequency = " << total_clk / total_time / 1e6
            << " (MHz)\n";

  // std::cout << "---------------------------------------------------------\n";

  // Free device memory

  cudaCheckError(cudaFree(d_startClk));
  cudaCheckError(cudaFree(d_stopClk));
  cudaCheckError(cudaFree(d_timeStart));
  cudaCheckError(cudaFree(d_timeStop));
  cudaCheckError(cudaFree(d_X));
  cudaCheckError(cudaFree(d_Y));

  // Free host memory

  free(startClk);
  free(stopClk);
  free(timeStart);
  free(stopStop);
  free(h_X);
  free(h_Y);

  return 0;
}
