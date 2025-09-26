#include "tensor.h"



// ==============================
// Factory: zeros
// ==============================
Tensor Tensor::zeros(const Shape& shape, Dtype dtype, const Device& device) {
    Tensor t(shape, dtype, device); // allocates memory
    if (dtype == Dtype::Float32) {
        float* ptr = t.data<float>();
        std::fill(ptr, ptr + t.numel(), 0.0f);
    } else if (dtype == Dtype::Int32) {
        int* ptr = t.data<int>();
        std::fill(ptr, ptr + t.numel(), 0);
    }
    return t;
}

// ==============================
// Factory: ones
// ==============================
Tensor Tensor::ones(const Shape& shape, Dtype dtype, const Device& device) {
    Tensor t(shape, dtype, device); // allocates memory
    if (dtype == Dtype::Float32) {
        float* ptr = t.data<float>();
        std::fill(ptr, ptr + t.numel(), 1.0f);
    } else if (dtype == Dtype::Int32) {
        int* ptr = t.data<int>();
        std::fill(ptr, ptr + t.numel(), 1);
    }
    return t;
}
