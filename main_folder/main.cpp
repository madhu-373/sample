#include "tensor.h"
#include<cstring>
using namespace std;
int main() {
    // Create a zeros tensor
    std::vector<std::vector<float>> data = {{0,-1,3},{1,2,3}, {4,5,6}};

    // Create tensor directly with data
    Tensor t(data, Dtype::Float32, Device(DeviceType::CPU));

    cout<<"Tensor t: ";
    // Access and print data
    t.print_info();
    float* d = t.data<float>();
    /*
    for (size_t i = 0; i < t.numel(); ++i) {
        cout << d[i] << " ";
    }*/
    int i = 1, j = 2;
    //float val = d[i * cols + j];
    cout<<"Rows:"<<endl;
    cout<<t.shape_.dims[0]<<endl;
    cout<<"Cols:"<<endl;
    cout<<t.shape_.dims[1]<<endl;
    cout<<"Element at ("<<i<<","<<j<<"):"<<endl;
    cout<<d[i * t.shape_.dims[1] + j]<<endl;
    cout<<endl;
    Tensor t1 = Tensor::zeros({2, 3});
    t1.print_info();
    // Create a ones tensor
    Tensor t2 = Tensor::ones({2, 3}, Dtype::Float32);

    
    cout << std::endl;
}
