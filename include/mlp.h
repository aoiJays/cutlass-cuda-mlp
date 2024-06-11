#pragma once

#include <vector>
#include <fstream>
#include <string>

#include "layer.h"

class MLP {

    private:
        std::vector< std::shared_ptr<Layer> > mlp;
        std::vector< std::string > mlp_info;

        int batch_size, inputDim, outputDim;
        bool requires_grad;
        std::ifstream in;
    public:

        MLP(int batch_size, int inputDim, std::string path_to_bin, bool requires_grad = false);
        
        void addPositionalEmbedding(int level); 
        
        void addLinear(int LayerDim); 

        void addReLU(); 
        void addSigmoid(); 
        
        Matrix * forward(Matrix * dataset);
        Matrix * autograd();
        
        int getOutputDim() const;
        void print() const;
        ~MLP();
};