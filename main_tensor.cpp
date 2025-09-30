#include "tensor.h"
#include <iostream>

int main() {
    try {
        //Shape s({0,0,0}); // Empty tensor raises error
        Shape s({2, 3, 4}); // 3D tensor
        Tensor t(s, Dtype::Float32);

        std::cout << "Num elements: " << t.numel() << "\n";
        std::cout << "Total bytes: " << t.nbytes() << "\n";

        for(int i=0;i< t.shape().size(); i++) {
            std::cout << "Dim " << i << ": " << t.shape()[i] << "\n";
        }

        //print all the elements of the tensor, using shape() and stride()
        for(int i=0;i<t.numel(); i++) {
            std::cout << "Element " << i << ": ";
            for(int j=0;j<t.shape().size(); j++) {
                int index = (i / t.stride()[j]) % t.shape()[j];
                std::cout << index << " ";
            }
            std::cout << "\n";
            
        }

        std::cout << "Shape: ";
        for (auto dim : t.shape()) std::cout << dim << " ";
        std::cout << "\nStride: ";
        for (auto st : t.stride()) std::cout << st << " ";
        std::cout << "\n";

        // Access data
        float* data = t.data<float>();
        data[0] = 3.14f;
        std::cout << "First element = " << data[0] << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
    return 0;
}
