#pragma once

#include <string>
#include <fstream>

#include "matrix.h"

class DataLoader {

    private:
        
        int inputDim, batch_size, data_size;
        float * data;
        int batch_pos;

    public: 
        DataLoader(int inputDim, int data_size, int batch_size, float * data );
        void load(Matrix * X);
        bool empty() const;
};
