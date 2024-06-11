#pragma once

#include "cutlass/gemm/device/gemm.h" 

/*  
    GPU Memory Pointers
    d_C(M,N) = d_A(M,K) * d_B(K,N)
    
*/

void cutlassMatrixMul(float * d_A, float * d_B, float * d_C, int M, int N, int K);
void cutlassMatrixMul_trans(float * d_A, float * d_B, float * d_C, int M, int N, int K);