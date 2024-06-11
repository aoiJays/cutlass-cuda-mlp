#include "layer.h"

#include "matrix.h"
#include "activation.h"

// Layer::Layer() {}
Layer::Layer(int batch_size, int LayerDim, int lastLayerDim, bool requires_grad) : LayerDim(LayerDim) , lastLayerDim(lastLayerDim), requires_grad(requires_grad){
    output = std::make_shared<Matrix>(LayerDim, batch_size);    
    if ( requires_grad ) grad = std::make_shared<Matrix>(lastLayerDim, batch_size);
}

int Layer::getLayerDim() const {
    return LayerDim;
}

Matrix* Layer::getAddr() const {
    return output.get();
}

Matrix* Layer::getGrad() const {
    return grad.get();
}

PositionalEmbedding::PositionalEmbedding(int batch_size, int LayerDim, int lastLayerDim, int level, bool requires_grad) : Layer(batch_size, LayerDim, lastLayerDim, requires_grad), level(level) {}
void PositionalEmbedding::forward(Matrix * input, int lastLayerDim, int batch_size) {
    Matrix::matrixPositionalEmbedding(output.get(), input, lastLayerDim, batch_size, level);
}
void PositionalEmbedding::backward(Matrix * input, int lastLayerDim, int batch_size) {}




Linear::Linear(int batch_size, int LayerDim, int lastLayerDim, float * _W, float * _b, bool requires_grad) : Layer(batch_size, LayerDim, lastLayerDim, requires_grad) {
    W = std::make_shared<Matrix>(LayerDim, lastLayerDim, _W);
    b = std::make_shared<Matrix>(LayerDim, 1, _b); 
}

void Linear::forward(Matrix * input, int lastLayerDim, int batch_size) {
    Matrix::matrixMul(W.get(), input, output.get(), LayerDim, batch_size, lastLayerDim);
    Matrix::matrixAdd(output.get(), b.get(), LayerDim, batch_size);
}

void Linear::backward(Matrix * input, int lastLayerDim, int batch_size) {
    Matrix::matrixMul_trans(W.get(), input, grad.get(), lastLayerDim, batch_size, LayerDim);
}




ReLU::ReLU(int batch_size, int LayerDim, int lastLayerDim, bool requires_grad) : Layer(batch_size, LayerDim, lastLayerDim, requires_grad) {}
void ReLU::forward(Matrix * input, int lastLayerDim, int batch_size) { 
    Matrix::matrixFunc(output.get(), input, F_ReLU, LayerDim, batch_size);
}
void ReLU::backward(Matrix * input, int lastLayerDim, int batch_size) {
    Matrix::matrixFunc_dot(grad.get(), input, F_dReLU, lastLayerDim, batch_size);
}



Sigmoid::Sigmoid(int batch_size, int LayerDim, int lastLayerDim, bool requires_grad) : Layer(batch_size, LayerDim, lastLayerDim, requires_grad) {}
void Sigmoid::forward(Matrix * input, int lastLayerDim, int batch_size) {
    Matrix::matrixFunc(output.get(), input, F_Sigmoid, LayerDim, batch_size);
}

void Sigmoid::backward(Matrix * input, int lastLayerDim, int batch_size) {
    Matrix::matrixFunc_dot(grad.get(), input, F_dSigmoid, lastLayerDim, batch_size);
}