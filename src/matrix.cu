#include "matrix.h"

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <random>
#include <cstdio>

#include "cuda_run.h"
#include "cutlass_matrix.h"

float matrix_gen() {
    static std::random_device rd;
    static std::mt19937 seed(rd()); 
    static std::uniform_real_distribution<float> dis(-1,1);
    return dis(seed);
}

Matrix::Matrix() : m(0), n(0) {
    gpu.reset(NULL);
}

Matrix::Matrix(int m, int n, float * p) : m(m), n(n) {
    cudacheck(cudaMalloc( (void**)&gpu, m * n *sizeof(float)));
    cudacheck(cudaMemcpy(gpu.get(), p, m * n * sizeof(float), cudaMemcpyHostToDevice));
}

Matrix::Matrix(int m, int n) : m(m), n(n) {

    int sz = m * n;
    float *p = new float[sz]; 

    srand(time(0));
    for (int i=0;i<sz;++i)
        p[i] = matrix_gen();

    cudacheck(cudaMalloc( (void**)&gpu, m * n *sizeof(float)));
    cudacheck(cudaMemcpy(gpu.get(), p, m * n * sizeof(float), cudaMemcpyHostToDevice));

    delete[] p;
}

Matrix::Matrix(int m, int n, int ones) : m(m), n(n) {

    int sz = m * n;
    float *p = new float[sz]; 

    srand(time(0));
    for (int i=0;i<sz;++i)
        p[i] = ones;

    cudacheck(cudaMalloc( (void**)&gpu, m * n *sizeof(float)));
    cudacheck(cudaMemcpy(gpu.get(), p, m * n * sizeof(float), cudaMemcpyHostToDevice));

    delete[] p;
}

void Matrix::MatrixCat(Matrix * a, Matrix * b) {

    if ( m != a->m + b->m ) {
        std::cerr << "Error : New matrix' s rows not equal to a.m + b.m" << std::endl;
        exit(-1);
    }

    if ( n != a->n || n != b->n ) {
        std::cerr << "Error : New matrix' s columns not equal to a.n or b.n" << std::endl;
        exit(-1);
    }

    cudacheck(cudaMemcpy(gpu.get(), a->gpu.get(), a->m * n * sizeof(float), cudaMemcpyDeviceToDevice));
    cudacheck(cudaMemcpy(gpu.get() + a->m * n , b->gpu.get(), b->m * n * sizeof(float), cudaMemcpyDeviceToDevice));
}

void Matrix::MatrixSet(int m, int n, float * p) {
    this->m = m; this->n = n; gpu.reset(NULL);
    cudacheck(cudaMalloc( (void**)&gpu, m * n *sizeof(float)));
    cudacheck(cudaMemcpy(gpu.get(), p, m * n * sizeof(float), cudaMemcpyHostToDevice));
}

void Matrix::print() {
    printf("_________(%d, %d)____________\n", m, n);
    float * cpu = new float[m * n];
    cudacheck(cudaMemcpy(cpu, gpu.get(), m * n * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i=0;i<m;++i)
        for (int j=0;j<n;++j) {
            printf("%.7f\t", cpu[i * n + j]);
            if (j==n-1) printf("\n");
        }
    printf("_____________________\n");
    delete[] cpu;
}

void Matrix::matrixMul(Matrix * A, Matrix * B, Matrix * C, int M, int N, int K) {
    cutlassMatrixMul(A->gpu.get(), B->gpu.get(), C->gpu.get(), M, N, K);
    cudaDeviceSynchronize();
} 

void Matrix::matrixMul_trans(Matrix * A, Matrix * B, Matrix * C, int M, int N, int K) {
    cutlassMatrixMul_trans(A->gpu.get(), B->gpu.get(), C->gpu.get(), M, N, K);
    cudaDeviceSynchronize();
} 


void Matrix::matrixAdd(Matrix * A, Matrix * b, int M, int N) {
    assert(A->m == b->m && b->n == 1);
    dim3 blockSize(256);
    dim3 gridSize((M * N + blockSize.x - 1) / blockSize.x);
	cu_matrixAdd<<<gridSize, blockSize>>>(A->gpu.get(), b->gpu.get(), M, N);
    cudaDeviceSynchronize();
} 

void Matrix::matrixFunc(Matrix * C, Matrix * A, Activataion func, int M, int N) {
    dim3 blockSize(256);
    dim3 gridSize((M * N + blockSize.x - 1) / blockSize.x);
    cu_matrixFunc<<<gridSize, blockSize>>>(C->gpu.get(), A->gpu.get(), func, M, N);
    cudaDeviceSynchronize();
}

void Matrix::matrixFunc_dot(Matrix * C, Matrix * A, Activataion func, int M, int N) {
    dim3 blockSize(256);
    dim3 gridSize((M * N + blockSize.x - 1) / blockSize.x);
    cu_matrixFunc_dot<<<gridSize, blockSize>>>(C->gpu.get(), A->gpu.get(), func, M, N);
    cudaDeviceSynchronize();
}

void Matrix::matrixPositionalEmbedding(Matrix* C, Matrix* A, int M, int N, int level) {
    dim3 blockSize(256);
    dim3 gridSize((M * N + blockSize.x - 1) / blockSize.x);
    cu_matrixPositionalEmbedding<<<gridSize, blockSize>>>(C->gpu.get(), A->gpu.get(), M, N, level);
    cudaDeviceSynchronize();
}