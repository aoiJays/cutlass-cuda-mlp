#include "cutlass_matrix.h"
#include <iostream>

void cutlassMatrixMul(float * d_A, float * d_B, float * d_C, int M, int N, int K) {
    
    using Gemm = cutlass::gemm::device::Gemm<
        float, cutlass::layout::RowMajor, // 数据类型和布局类型 A
        float, cutlass::layout::RowMajor, // 数据类型和布局类型 B
        float, cutlass::layout::RowMajor, // 数据类型和布局类型 C
        float                             // 累加器类型
    >;

    // 创建 CUTLASS GEMM 运算对象
    Gemm gemm_op;

    // 设置 GEMM 操作参数
    cutlass::gemm::GemmCoord problem_size(M, N, K);
    float alpha = 1.0f;
    float beta = 0.0f;

    // 定义 CUTLASS GEMM 操作参数
    typename Gemm::Arguments arguments{
        problem_size,
        {d_A, K},
        {d_B, N},
        {d_C, N},
        {d_C, N},
        {alpha, beta},
    };

    // 检查参数合法性
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "GEMM operation cannot be implemented: " << cutlass::cutlassGetStatusString(status) << std::endl;
        exit(-1);
    }

    // 执行 GEMM 运算
    status = gemm_op(arguments);

    // 检查运算状态
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "GEMM operation failed: " << cutlass::cutlassGetStatusString(status) << std::endl;
        exit(-1);
    }
}     
