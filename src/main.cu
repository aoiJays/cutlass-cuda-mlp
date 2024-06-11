#include <iostream>
#include <chrono>

#include "dataloader.h"
#include "mlp.h"
#include "matrix.h"

int main() {

    // init 3d Points
    float * x = new float[256];
    double gap = (1 - (-1) ) / (double)255;

    for (int i=0;i<256;++i)
        x[i] = -1 + i * gap;

    float * pts = new float[256 * 256 * 256 * 2];
    float * pts3d = new float[256 * 256 * 256 * 3];


    // init 2d Points
    auto make_2d_pts = [&](int dir) {
        int cnt = 0;
        for (int i=0;i<256;++i)
            for (int j=0;j<256;++j)
                for (int k=0;k<256;++k)  
                    if ( dir == 2 ) {
                        pts[cnt] = x[i], pts[cnt + 1] = x[j], cnt += 2;
                    }
                    else if ( dir == 1 ) {
                        pts[cnt] = x[i], pts[cnt + 1] = x[k], cnt += 2;
                    }
                    else{
                        pts[cnt] = x[j], pts[cnt + 1] = x[k], cnt += 2;
                    }
    };

    // init 2d Points
    auto make_3d_pts = [&]() {
        int cnt = 0;
        for (int i=0;i<256;++i)
            for (int j=0;j<256;++j)
                for (int k=0;k<256;++k)  
                        pts3d[cnt] = x[i], pts3d[cnt + 1] = x[j], pts3d[cnt + 2] = x[k], cnt += 3;

    };


    const int data_size = 256 * 256 * 256;
    const int batch_size = 65536;
    const int inputDim = 2;

  
    MLP mask_net(batch_size, inputDim, "../model/mask_model_weight.data");
    
    mask_net.addPositionalEmbedding(10);
    mask_net.addLinear(256);
    mask_net.addReLU();
    mask_net.addLinear(256);
    mask_net.addReLU();
    mask_net.addLinear(1);
    mask_net.addSigmoid();

    mask_net.print();

    Matrix X(inputDim, batch_size);

    for (int dir=2;dir>=0;--dir) {
        make_2d_pts(dir);
        DataLoader dataLoader(inputDim, data_size, batch_size, pts);

        std::chrono::duration<double> sum(0);

        while ( !dataLoader.empty() ) {
            dataLoader.load(&X); // load data

            auto start = std::chrono::high_resolution_clock::now();
            Matrix * res = mask_net.forward(&X);
            auto end = std::chrono::high_resolution_clock::now();

            auto duration = end - start;
            sum += duration;
            // res->print();
        }

        std::cout << "mask总运行时间: " << sum.count() << " 秒" << std::endl; 
    }
  

    MLP preEmbedding_net(batch_size, inputDim, "");
    preEmbedding_net.addPositionalEmbedding(10);

    preEmbedding_net.print();

    MLP net_with_grad(batch_size, preEmbedding_net.getOutputDim() + 3, "../model/net_model_weight.data", true);
    
    net_with_grad.addLinear(256);

    net_with_grad.addReLU();

    net_with_grad.addLinear(256);
    net_with_grad.addReLU();

    net_with_grad.addLinear(256);
    net_with_grad.addReLU();

    net_with_grad.addLinear(256);
    net_with_grad.addReLU();

    net_with_grad.addLinear(256);
    net_with_grad.addReLU();

    net_with_grad.addLinear(256);
    net_with_grad.addReLU();
    
    net_with_grad.addLinear(256);
    net_with_grad.addReLU();

    net_with_grad.addLinear(256);
    net_with_grad.addReLU();

    net_with_grad.addLinear(256);
    net_with_grad.addReLU();

    net_with_grad.addLinear(1);

    net_with_grad.print();

      
    Matrix X_2d(inputDim, batch_size);
    Matrix X_3d(3, batch_size);

    
    Matrix inputs_pts(3 + preEmbedding_net.getOutputDim(), batch_size);

    make_3d_pts();

    
    for (int dir=2;dir>=0;--dir) {

        make_2d_pts(dir);
        DataLoader dataLoader(inputDim, data_size, batch_size, pts);
        DataLoader dataLoader3D(3, data_size, batch_size, pts3d);

        std::chrono::duration<double> sum(0);

        while ( !dataLoader.empty() ) {
            
            dataLoader.load(&X_2d); 
            Matrix * res_2d = preEmbedding_net.forward(&X_2d);

            dataLoader3D.load(&X_3d); 
            inputs_pts.MatrixCat(&X_3d, res_2d);

            auto start = std::chrono::high_resolution_clock::now();
            Matrix * pred = net_with_grad.forward(&inputs_pts);
            auto end = std::chrono::high_resolution_clock::now();

            auto duration = end - start;
            sum += duration;
            // pred->print();

            // something bad …… need more time
            // Matrix * grad = net_with_grad.autograd();
            // grad->print();
            // break;            
        }

        std::cout << "总运行时间: " << sum.count() << " 秒" << std::endl; 
    }

    delete[] x;
    delete[] pts;
}