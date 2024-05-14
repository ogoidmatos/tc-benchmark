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
#define K 8

#define THREADS_PER_BLOCK 1024
#define NUM_BLOCKS 32768
#define A_SIZE M *K *(THREADS_PER_BLOCK / 32) * NUM_BLOCKS
#define B_SIZE K *N *(THREADS_PER_BLOCK / 32) * NUM_BLOCKS
#define C_SIZE M *N *(THREADS_PER_BLOCK / 32) * NUM_BLOCKS
#define ITERATIONS 32768 / 2

#define MEM 1
#define FLOP 16

#if MEM > FLOP
#define REMAINDER MEM - FLOP
#elif FLOP > MEM
#define REMAINDER FLOP - MEM
#endif

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
__global__ void benchmark_alt(float *d_A, float *d_B, float *d_C, float *d_X,
                              float *d_Y, uint64_t *d_startClk,
                              uint64_t *d_stopClk, uint64_t *d_timeStart,
                              uint64_t *d_timeStop) {
  // Code to be executed on the GPU
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t start = 0;
  uint64_t stop = 0;
  uint64_t time_start = 0;
  uint64_t time_stop = 0;

  // create registers for threads
  float fragsA[4];
  float fragsB[2];
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

  // start timing
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(time_start)::"memory");
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(start)::"memory");

  for (int i = 0; i < ITERATIONS; i++) {
#if MEM == 1 || FLOP == 1
#pragma unroll
    for (int j = 0; j < MEM; j++) {
      // SWITCH ORDER HERE TO MAKE SURE THAT C0 IS WRITTEN TO MEMORY BEFORE
      // BEING OVERWRITTEN; NECESSARY FOR COMPILER TO NOT OPTIMIZE AWAY THE
      // COMPUTATION
      d_Y[id] = fragsC[0];
      fragsC[0] = d_X[id];
    }
#pragma unroll
    for (int j = 0; j < FLOP; j++) {
      // assembly mma
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]));
    }
#elif MEM > FLOP
#pragma unroll
    for (int j = 0; j < FLOP; j++) {
      d_Y[id] = fragsC[0];
      fragsC[0] = d_X[id];
      // assembly mma
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]));
    }
#pragma unroll
    for (int j = 0; j < REMAINDER; j++) {
      d_Y[id] = fragsC[0];
      fragsC[0] = d_X[id];
    }
#else
#pragma unroll
    for (int j = 0; j < MEM; j++) {
      d_Y[id] = fragsC[0];
      fragsC[0] = d_X[id];
      // assembly mma
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]));
    }
#pragma unroll
    for (int j = 0; j < REMAINDER; j++) {
      asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
          "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
          : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])
          : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]));
    }
#endif
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
  // d_X[id] = fragsC[0];
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

  // handle l2 mem handles
  float *h_X = (float *)malloc(NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(float));
  float *h_Y = (float *)malloc(NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(float));
  float *d_X, *d_Y;
  cudaCheckError(cudaMalloc((void **)&d_X,
                            NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(float)));
  cudaCheckError(cudaMalloc((void **)&d_Y,
                            NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(float)));
  cudaCheckError(cudaMemcpy(d_X, h_X,
                            NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(float),
                            cudaMemcpyHostToDevice));
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

  // Prepare
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  // Start record
  cudaEventRecord(start, 0);

  thread_args.flag = 1;
  // Launch kernel on the GPU
  benchmark_alt<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(
      d_A, d_B, d_C, d_X, d_Y, d_startClk, d_stopClk, d_timeStart, d_timeStop);

  // Stop event
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, start, stop);  // that's our time!
  printf("Elapsed time: %f ms\n", elapsedTime);
  // Clean up:
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

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

  std::cout << "Bandwidth = " << bw << " (bytes/sec)\n";

  std::cout << "FLOPS = " << FLOPS << "(TFLOPs) \n";
  std::cout << "AI = " << fma * 2. / bytes << " (FLOP/byte)\n";

  std::cout << "Total Clk number = " << total_clk << "\n";

  std::cout << "Total Time number = " << total_time << " (sec)\n";
  std::cout << "Average Clock Frequency = " << total_clk / total_time / 1e6
            << " (MHz)\n";
  std::cout << bytes << " bytes\n";

  std::cout << "---------------------------------------------------------\n";

  // Free device memory
  cudaCheckError(cudaFree(d_A));
  cudaCheckError(cudaFree(d_B));
  cudaCheckError(cudaFree(d_C));
  cudaCheckError(cudaFree(d_X));
  cudaCheckError(cudaFree(d_Y));
  cudaCheckError(cudaFree(d_startClk));
  cudaCheckError(cudaFree(d_stopClk));
  cudaCheckError(cudaFree(d_timeStart));
  cudaCheckError(cudaFree(d_timeStop));

  // Free host memory
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_X);
  free(h_Y);
  free(startClk);
  free(stopClk);
  free(timeStart);
  free(stopStop);

  return 0;
}
