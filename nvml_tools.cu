#include <nvml.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>

typedef struct {
  std::vector<int> powerArray;
  std::vector<int> clockArray;
  nvmlDevice_t device;
  int flag;
  bool verbose;
} monitor_args;

void monitoring(monitor_args* args) {
  nvmlReturn_t result;

  unsigned int power;
  unsigned int clockSM;

  nvmlDevice_t device;
  result = nvmlDeviceGetHandleByIndex(0, &device);
  if (NVML_SUCCESS != result) {
    printf("Failed to get handle for device 0: %s\n", nvmlErrorString(result));
    exit(1);
  }

  // waiting to start measuring
  while (args->flag == 0) {
  }

  while (args->flag) {
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

    (args->powerArray).push_back(power);
    (args->clockArray).push_back(clockSM);
    if (args->verbose) {
      printf("Power: %.3f W  ", (float)power / 1000.0f);
      printf("Clock: %d MHz\n", clockSM);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return;
}

void init_nvml(monitor_args* thread_args, std::thread* measuring_thread,
               bool verbose = true) {
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

  thread_args->device = device;
  thread_args->verbose = verbose;

  *measuring_thread = std::thread(monitoring, thread_args);
  return;
}

void stop_nvml(std::thread* measuring_thread, std::vector<int> powerArray,
               std::vector<int> clockArray) {
  measuring_thread->join();
  nvmlReturn_t result;
  result = nvmlShutdown();
  if (NVML_SUCCESS != result) {
    printf("Failed to shutdown NVML: %s\n", nvmlErrorString(result));
    exit(1);
  }
  // write results to file
  std::ofstream statsFile;
  statsFile.open("power.csv");
  statsFile << "Time;Power;Clock; " << std::endl;
  for (int i = 0; i < powerArray.size(); i++) {
    statsFile << (float)i * 10.0 / 1000 << ";" << (float)powerArray[i] / 1000.0f
              << ";" << clockArray[i] << ";" << std::endl;
  }
  statsFile.close();
  return;
}