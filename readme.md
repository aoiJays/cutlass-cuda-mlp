## demo

```bash
cd build
cmake ..
make -j
./app
```

```cpp
// main.cu
#include <iostream>
#include <chrono>

#include "dataloader.h"
#include "mlp.h"
#include "matrix.h"

int main() {


    // prepare for inputs

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

    // 初始化：模型参数（用python事先导出的
    MLP mask_net(batch_size, inputDim, "../model/mask_model_weight.data");
    
    mask_net.addPositionalEmbedding(10);
    mask_net.addLinear(256);
    mask_net.addReLU();
    mask_net.addLinear(256);
    mask_net.addReLU();
    mask_net.addLinear(1);
    mask_net.addSigmoid();

    // 模型预览
    mask_net.print();

    Matrix X(inputDim, batch_size);

    for (int dir=2;dir>=0;--dir) {
        
        make_2d_pts(dir); // 准备数据

        // 嵌入数据 （每个batch_size轮流进入GPU） // noted: 似乎之后需要整个塞进去
        DataLoader dataLoader(inputDim, data_size, batch_size, pts);

        std::chrono::duration<double> sum(0);

        while ( !dataLoader.empty() ) {
            dataLoader.load(&X); // load data

            auto start = std::chrono::high_resolution_clock::now();
            Matrix * res = mask_net.forward(&X); // 推理
            auto end = std::chrono::high_resolution_clock::now();

            auto duration = end - start;
            sum += duration;
            // res->print(); // 输出矩阵不大的可以输出一下
        }

        std::cout << "mask总运行时间: " << sum.count() << " 秒" << std::endl; 
    }
  


    delete[] x;
    delete[] pts;
}
```

## .model

先做的模型参数从文件中进行读取

文件格式：

```plain
w11 w12 w13 ... w21 w22 w23 ... wn1 wn2 wn3
b1 b2 b3 b4 ...
w11 w12 w13 ... w21 w22 w23 ... wn1 wn2 wn3
b1 b2 b3 b4 ...
w11 w12 w13 ... w21 w22 w23 ... wn1 wn2 wn3
b1 b2 b3 b4 ...
```

目前支持Linear层读取W和b
