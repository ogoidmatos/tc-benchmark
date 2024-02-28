#include <nvml.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <thread>
// #include <vector>
#include <thrust/host_vector.h>

typedef struct {
  thrust::host_vector<int> powerArray;
  thrust::host_vector<int> clockArray;
  nvmlDevice_t device;
  int* flag;
} monitor_args;

void init_nvml(int* flag, std::thread* measuring_thread,
               thrust::host_vector<int>* powerArray,
               thrust::host_vector<int>* clockArray) {
  nvmlReturn_t result;
  result = nvmlInit();
  if (NVML_SUCCESS != result) {
    printf("Failed to initialize NVML: %s\n", nvmlErrorString(result));
    exit(1);
  }

  nvmlDevice_t device;
  result = nvmlDeviceGetHandleByIndex(0, &device);
  if (NVML_SUCCESS != result) {
    printf("Failed to get handle for device 0: %s\n", nvmlErrorString(result));
    exit(1);
  }

  monitor_args thread_args;
  thread_args.device = device;
  thread_args.powerArray = powerArray;
  thread_args.clockArray = clockArray;
  thread_args.flag = flag;

  *measuring_thread = std::thread(monitoring, &thread_args);
  return;
}

void monitoring(monitor_args* args) {
  nvmlReturn_t result;

  unsigned int power;
  unsigned int clockSM;

  // waiting to start measuring
  while (args->*flag == 0) {
  }

  while (args->*flag) {
    power = 0;
    clockSM = 0;
    result = nvmlDeviceGetPowerUsage(device, &power);
    if (NVML_ERROR_NOT_SUPPORTED == result)
      printf("This does not support power measurement\n");
    else if (NVML_SUCCESS != result) {
      printf("Failed to get power for device %i: %s\n", 0,
             nvmlErrorString(result));
      exit(1);
    }

    result = nvmlDeviceGetClock(device, NVML_CLOCK_SM, NVML_CLOCK_ID_CURRENT,
                                &clockSM);
    if (NVML_ERROR_NOT_SUPPORTED == result)
      printf("This does not support clock measurement\n");
    else if (NVML_SUCCESS != result) {
      printf("Failed to get SM clock for device 0: %s\n",
             nvmlErrorString(result));
      exit(1);
    }

    (args->*powerArray).push_back(power / 1000);
    (args->*clockArray).push_back(clockSM);
    printf("Power: %d mw\n", power / 1000);
    printf("Clock: %d MHz\n", clockSM);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return;
}

void stop_nvml(std::thread* measuring_thread,
               thrust::host_vector<int>* powerArray,
               thrust::host_vector<int>* clockArray) {
  measuring_thread->join();
  nvmlReturn_t result;
  result = nvmlShutdown();
  if (NVML_SUCCESS != result) {
    printf("Failed to shutdown NVML: %s\n", nvmlErrorString(result));
    exit(1);
  }
  // write results to file
  std::ofstream statsFile;
  statsFile.open("power.txt");
  for (int i = 0; i < *powerArray.size(); i++) {
    statsFile << i * 10 / 1000 << " " << powerArray[i] << " " << clockArray[i]
              << std::endl;
  }
  statsFile.close();
  return;
}