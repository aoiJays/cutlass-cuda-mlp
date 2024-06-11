#include "cuda_run.h"
#include "activation.h"

__global__ void cu_matrixAdd(float * dst, float * src, int m, int n) {
	int tx = threadIdx.x, bx = blockIdx.x;
	int id = bx * blockDim.x + tx;
	if ( id >= m * n ) return;
	int x = id / n;
	dst[id] += src[ x ];
}

__global__ void cu_matrixFunc(float * dst, float * src, Activataion func, int m, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m*n) dst[idx] = Func(src[idx], func);
}

__global__ void cu_matrixPositionalEmbedding(float* d_C, float* d_A, int M, int N, int level) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        int row = idx / N;
        int col = idx % N;
        
        int pos = row * N + col;
        d_C[pos] = d_A[idx]; pos += M * N;
        
        float powf2 = 1.0;
        for (int l = 0; l < level; ++l, powf2 *= 2) {
            float val = d_A[idx] * powf2;
            d_C[pos] = sinf(val); d_C[pos + M * N] = cosf(val);
            pos += 2 * M * N;
        }
    }
}