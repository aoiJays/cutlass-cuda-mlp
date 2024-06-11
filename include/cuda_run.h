#pragma once

#include <cuda_runtime.h>
#include <iostream>

#include "activation.h"

// CUDA 错误检查宏
#define cudacheck(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void cu_matrixAdd(float * dst, float * src, int m, int n);
__global__ void cu_matrixFunc(float * dst, float * src, Activataion func, int m, int n);
__global__ void cu_matrixPositionalEmbedding(float* d_C, float* d_A, int M, int N, int level);
