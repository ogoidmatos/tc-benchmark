# tc-benchmark

Code for Tensor Core Micro Benchmarking, for PIC2 MEEC

## Current Tasks

- [ ] Half Precision
  - [x] FMA benchmark
  - [ ] Compute Roof -> FLOPs
- [ ] Alternate Formats
  - [ ] Bf16
  - [ ] Tf32
- [ ] Single Precision
- [ ] Double Precision
- [ ] Integer

## Table

| Inputs | Accumulators |  Shape   | FMA/clk/SM | FLOPs |
| ------ | ------------ | -------- | ---------- | ----- |
| FP16   | FP16         | m16n8k8  | 511.438    |       |
| FP16   | FP16         | m16n8k16 | 511.820    |       |
| FP16   | FP16         | m8n8k4ª  | 1.0093     |       |
| FP16   | FP32         | m16n8k8  | 255.675    |       |
| FP16   | FP32         | m16n8k16 | 255.942    |       |
| FP16   | FP32         | m8n8k4ª  | 1.0441     |       |

*Notes:* ª - "`mma.sync.m8n8k4` is optimized for target architecture `sm_70` and may have substantially reduced performance on other target architectures." [1](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=alternate#multiply-and-accumulate-instruction-mma)
