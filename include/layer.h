#pragma once

#include <memory>
#include "matrix.h"

class Layer {
    
    protected:
        std::shared_ptr<Matrix> output;
        std::shared_ptr<Matrix> grad;
        
        int LayerDim, lastLayerDim;
        bool requires_grad;

    public:
        
        virtual void forward(Matrix * input, int lastLayerDim, int batch_size) = 0;
        virtual void backward(Matrix * input, int lastLayerDim, int batch_size) = 0;
        Layer(int batch_size, int LayerDim, int lastLayerDim, bool requires_grad);

        int getLayerDim() const;
        Matrix * getAddr() const;
        Matrix * getGrad() const;
};

class PositionalEmbedding : public Layer {

    private:
        int level;
    public:
        PositionalEmbedding(int batch_size, int LayerDim, int lastLayerDim, int level, bool requires_grad);
        void forward(Matrix * input, int lastLayerDim, int batch_size);
        void backward(Matrix * input, int lastLayerDim, int batch_size);
};


class Linear : public Layer {

    private:
        std::shared_ptr<Matrix> W, b;

    public:
        Linear(int batch_size, int LayerDim, int lastLayerDim, float * _W, float * _b, bool requires_grad);
        void forward(Matrix * input, int lastLayerDim, int batch_size);
        void backward(Matrix * input, int lastLayerDim, int batch_size);
};

class ReLU : public Layer {

    private:

    public:
        ReLU(int batch_size, int LayerDim, int lastLayerDim, bool requires_grad);
        void forward(Matrix * input, int lastLayerDim, int batch_size);
        void backward(Matrix * input, int lastLayerDim, int batch_size);
};

class Sigmoid : public Layer {

    private:

    public:
        Sigmoid(int batch_size, int LayerDim, int lastLayerDim, bool requires_grad);
        void forward(Matrix * input, int lastLayerDim, int batch_size);
        void backward(Matrix * input, int lastLayerDim, int batch_size);
};

