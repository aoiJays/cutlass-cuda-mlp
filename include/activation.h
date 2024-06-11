#pragma once

typedef int Activataion;

#define F_ReLU 1
#define F_Sigmoid 2

#define F_dReLU 11
#define F_dSigmoid 12

#define max_float(a, b) (a > b ? a : b)

#define Func(x, func) \
    ((func) == F_ReLU ? max_float((x), 0.0f) : \
    (func) == F_Sigmoid ? (1 / (1 + std::exp(-(x)))) : \
    (func) == F_dReLU ? ((x) > 0 ? 1.0f : 0.0f) : \
    (func) == F_dSigmoid ? (std::exp(-(x)) / ((1 + std::exp(-(x))) * (1 + std::exp(-(x))))) : \
    ((x) > 0 ? (x) : 0.01f * (x)))
