#pragma once

#include <memory>
#include "cuda_run.h"
#include "activation.h"

// 自定义删除器，用于释放CUDA显存
struct CudaDeleter {
    void operator()(float* ptr) const {
        if (ptr==NULL) return;
        cudacheck(cudaFree(ptr));
    }
};

class Matrix {
    private:

        int m, n;
        std::unique_ptr<float, CudaDeleter> gpu;

    public:

        Matrix();
        Matrix(int m, int n);
        Matrix(int m, int n, float * p);
        Matrix(int m, int n, int ones);
        
        void MatrixSet(int m, int n, float * p);
        void MatrixCat(Matrix * a, Matrix * b);
        
        void print();

        static void matrixMul(Matrix * A, Matrix * B, Matrix * C, int M, int N, int K);
        static void matrixMul_trans(Matrix * A, Matrix * B, Matrix * C, int M, int N, int K);
        static void matrixAdd(Matrix * A, Matrix * b, int M, int N);
        static void matrixFunc(Matrix * C, Matrix * A, Activataion func, int M, int N);
        static void matrixFunc_dot(Matrix * C, Matrix * A, Activataion func, int M, int N);
        static void matrixPositionalEmbedding(Matrix * C, Matrix * A, int M, int N, int level);
};
