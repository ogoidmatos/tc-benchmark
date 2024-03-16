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
#define NUM_BLOCKS 1
#define ITERATIONS 32768L
#define SHARED_MEM_SIZE (32 * 1024 / 4)

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
__global__ void benchmark_alt(int *d_X, uint64_t *d_startClk,
                              uint64_t *d_stopClk, uint64_t *d_timeStart,
                              uint64_t *d_timeStop) {
  // Code to be executed on the GPU
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t start = 0;
  uint64_t stop = 0;
  uint64_t time_start = 0;
  uint64_t time_stop = 0;

  // __shared__ unsigned s[SHARED_MEM_SIZE];  // static shared memory

  // // one thread to initialize the pointer-chasing array
  // if (id == 0) {
  //   for (int i = 0; i < SHARED_MEM_SIZE; i++) s[i] = i * 16;
  // }
  // // synchronize threads
  // asm volatile("bar.sync 0;");

  // unsigned addr =
  //     static_cast<unsigned>(__cvta_generic_to_shared(&s[threadIdx.x * 4]));
  // https://stackoverflow.com/questions/76992939/confusion-about-cvta-generic-to-shared
  int x = 0;
  // synchronize threads
  asm volatile("bar.sync 0;");

  // start timing
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(time_start)::"memory");
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");

  for (int i = 0; i < ITERATIONS; i++) {
    // printf("addr = %d\n", addr);
    //  assembly mma
    asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];"
                 : "=r"(x)
                 : "r"(x));
  }
  // printf("x = %d\n", x);
  // for (int i = 0; i < ITERATIONS; i++) {
  //   //  assembly mma
  //   if (id == 0) {
  //     printf("addr s = %p\n", &s[threadIdx.x * 4]);
  //   }
  // }
  // // stop timing
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(stop)::"memory");
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(time_stop)::"memory");

  d_startClk[id] = start;
  d_stopClk[id] = stop;
  d_timeStart[id] = time_start;
  d_timeStop[id] = time_stop;
  d_X[id] = x;
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

  int *h_X = (int *)malloc(NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(int));
  int *d_X;
  cudaCheckError(
      cudaMalloc((void **)&d_X, NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(int)));
  cudaCheckError(cudaMemcpy(d_X, h_X,
                            NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(int),
                            cudaMemcpyHostToDevice));

  // // Allocate device memory
  // half *d_A, *d_B, *d_C;
  // cudaCheckError(cudaMalloc((void **)&d_A, dimA * sizeof(half)));
  // cudaCheckError(cudaMalloc((void **)&d_B, dimB * sizeof(half)));
  // cudaCheckError(cudaMalloc((void **)&d_C, dimC * sizeof(half)));

  // // Copy host memory to device
  // cudaCheckError(
  //     cudaMemcpy(d_A, h_A, dimA * sizeof(half), cudaMemcpyHostToDevice));
  // cudaCheckError(
  //     cudaMemcpy(d_B, h_B, dimB * sizeof(half), cudaMemcpyHostToDevice));
  // cudaCheckError(
  //     cudaMemcpy(d_C, h_C, dimC * sizeof(half), cudaMemcpyHostToDevice));

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
  benchmark_alt<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_X, d_startClk, d_stopClk,
                                                   d_timeStart, d_timeStop);

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

  long bytes = 8 * 8 * 2 * ITERATIONS * (THREADS_PER_BLOCK / 32) * NUM_BLOCKS;

  // uint64_t fma =
  //     (uint64_t)M * N * K * ITERATIONS * (THREADS_PER_BLOCK / 32) *
  //     NUM_BLOCKS;
  float bw = (float)bytes / (float)total_clk;

  // double FLOPS = fma * 2 / total_time / 1e12;

  // std::cout << "mma.sp.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16  latency
  // "
  //           << (float)total_clk / (float)ITERATIONS << " cycles\n";
  // std::cout
  //     << "mma.sp.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16  FMA Count "
  //     << fma << "\n";
  std::cout << "FMA tensor bandwidth = " << bw << " (FMA/clk/SM)\n";

  std::cout << "Total Clk number = " << total_clk << "\n";

  std::cout << "Total Time number = " << total_time << " (sec)\n";
  std::cout << "Average Clock Frequency = " << total_clk / total_time / 1e6
            << " (MHz)\n";
  // std::cout << "FLOPS = " << FLOPS << "(TFLOPs) \n";

  // std::cout << "---------------------------------------------------------\n";

  // Free device memory

  cudaCheckError(cudaFree(d_startClk));
  cudaCheckError(cudaFree(d_stopClk));
  cudaCheckError(cudaFree(d_timeStart));
  cudaCheckError(cudaFree(d_timeStop));

  // Free host memory

  free(startClk);
  free(stopClk);
  free(timeStart);
  free(stopStop);

  return 0;
}
