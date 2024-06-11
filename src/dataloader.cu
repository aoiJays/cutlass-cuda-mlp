#include <iostream>
#include <cstdlib>

#include "dataloader.h"

DataLoader::DataLoader(int inputDim, int data_size, int batch_size, float *data) : 
    inputDim(inputDim), data_size(data_size), batch_size(batch_size), data(data), batch_pos(0) {
        if ( data_size % batch_size != 0 ) {
            std::cerr << "Error : data_size MOD batch_size not equal to 0." << std::endl;
            exit(-1);
        }
    }

void DataLoader::load(Matrix * X) {
    float * temp = new float[batch_size * inputDim];

    // (a1,a2,a3), (b1,b2,b3), ... , (x1,x2,x3)
    // transform to 
    // (a1,b1,..., x1), (a2,b2,..., x2), (a3,b3,..., x3)

    for (int i=0;i<batch_size;++i) 
        for (int j=0;j<inputDim;++j)    
            temp[ j * batch_size + i ] = data[ i * inputDim + j ];

    X->MatrixSet(inputDim, batch_size, temp);

    batch_pos += batch_size;
    delete[] temp;


    batch_pos = data_size;
}

bool DataLoader::empty() const {
    return batch_pos >= data_size;
}