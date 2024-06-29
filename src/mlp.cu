#include "mlp.h"
#include "layer.h"

MLP::MLP(int batch_size, int inputDim, std::string path_to_bin, bool requires_grad) : batch_size(batch_size), inputDim(inputDim), outputDim(inputDim), requires_grad(requires_grad) {
    mlp.clear(); in.open(path_to_bin);
}


void MLP::addPositionalEmbedding(int level) {
    mlp.push_back( std::make_shared<PositionalEmbedding>(batch_size,outputDim * level * 2 + outputDim, outputDim, level, requires_grad) );
    mlp_info.push_back("PositionalEmbedding");
    outputDim =  outputDim * level * 2 + outputDim;
}


void MLP::addLinear(int LayerDim) {
    int lastLayerDim = outputDim;
    int sz = LayerDim * lastLayerDim;

    float * _W = new float[sz];
    for (int i=0;i<sz;++i) in >> _W[i];

    float * _b = new float[LayerDim];
    for (int i=0;i<LayerDim;++i) in >> _b[i];

    mlp.push_back( std::make_shared<Linear>(batch_size, LayerDim, lastLayerDim, _W, _b, requires_grad) );

    delete[] _W; delete[] _b;

    mlp_info.push_back("Linear");
    outputDim = LayerDim;
}

void MLP::addReLU() {
    int LayerDim = outputDim;
    mlp.push_back( std::make_shared<ReLU>(batch_size, LayerDim, outputDim, requires_grad) );
    mlp_info.push_back("ReLU");
}

void MLP::addSigmoid() {
    int LayerDim = outputDim;
    mlp.push_back( std::make_shared<Sigmoid>(batch_size, LayerDim, outputDim, requires_grad) );
    mlp_info.push_back("Sigmoid");
}

Matrix* MLP::forward(Matrix * dataset) {

    Matrix * input = dataset;
    int lastDim = inputDim;
    for (auto &it : mlp) {
        it->forward( input, lastDim, batch_size );
        input = it->getAddr();
        lastDim = it->getLayerDim();
    }
    return input;
}

Matrix * MLP::autograd() {

    if ( mlp.empty() ) {
        std::cerr << "Error: Empty MLP" << std::endl;
        exit(-1);
    }

    static Matrix X(outputDim, batch_size, 1);
    Matrix * input = &X;

    for (int i=(int)mlp.size() - 1;i>0;--i) {

        mlp[i]->backward(input, mlp[i-1]->getLayerDim(), batch_size);
        input = mlp[i]->getGrad();
    }

    mlp[0]->backward(input, inputDim, batch_size);
    return mlp[0]->getGrad();
}

int MLP::getOutputDim() const {
    return outputDim;
}


void MLP::print() const {

    std::cout << "=========== Network ============" << std::endl;
    for (int i=0;i<mlp.size();++i) {
        std::cout << mlp_info[i] << ": ";
        if ( mlp_info[i] == "InputLayer" || mlp_info[i] == "Linear" ) 
            std::cout << "LayerDim = "<< mlp[i]->getLayerDim() << "\n";
        else if (mlp_info[i] == "PositionalEmbedding")
            std::cout << "Embedding LayerDim = "<< mlp[i]->getLayerDim() << "\n";
        else std::cout << "Activation - LayerDim = "<< mlp[i]->getLayerDim() << "\n";
    }
    std::cout << "===============================" << std::endl;
}

MLP::~MLP() {
    in.close();
}