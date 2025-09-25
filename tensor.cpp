#include "tensor.h"
#include <iostream>
#include <algorithm>
#include <numeric>

// ========================================
// Helper: Compute total number of elements
// ========================================
size_t Tensor::numel() const {
    if (shape_.dims.empty()) {
        return 0;
    }
    return std::accumulate(shape_.dims.begin(), shape_.dims.end(),
                           static_cast<size_t>(1),
                           std::multiplies<size_t>());
}

// ========================================
// Helper: Compute strides for C-contiguous layout
// ========================================
void Tensor::compute_strides() {
    if (shape_.dims.empty()) {
        stride_.strides.clear();
        return;
    }

    stride_.strides.resize(shape_.dims.size());
    int32_t stride_val = 1;

    // Compute in reverse order for row-major layout
    for (int i = static_cast<int>(shape_.dims.size()) - 1; i >= 0; --i) {
        stride_.strides[i] = stride_val;
        stride_val *= shape_.dims[i];
    }
}

// ========================================
// Helper: Allocate memory based on device
// ========================================
void Tensor::allocate_memory() {
    const size_t total_bytes = nbytes();

    if (total_bytes == 0) {
        throw std::runtime_error("Cannot allocate memory for empty tensor.");
    }

    if (device_.type == DeviceType::CPU) {
        // CPU allocation using new + smart pointer
        data_ = std::shared_ptr<uint8_t>(
            new uint8_t[total_bytes],
            std::default_delete<uint8_t[]>()
        );
    }
    else if (device_.type == DeviceType::CUDA) {
        // Future GPU support
        // Example (pseudo-code):
        // void* gpu_ptr = nullptr;
        // cudaMalloc(&gpu_ptr, total_bytes);
        // data_ = std::shared_ptr<uint8_t>(
        //     static_cast<uint8_t*>(gpu_ptr),
        //     [](uint8_t* ptr) { cudaFree(ptr); }
        // );
        throw std::runtime_error("CUDA device allocation not implemented yet.");
    }
    else {
        throw std::runtime_error("Unsupported device type for allocation.");
    }

    is_owner_ = true;
}

// ========================================
// Tensor Constructor
// ========================================
Tensor::Tensor(const Shape& shape, Dtype dtype, const Device& device, bool requires_grad)
    : shape_(shape),
      dtype_(dtype),
      device_(device),
      requires_grad_(requires_grad),
      is_owner_(false) 
{
    // Validate shape
    if (shape_.dims.empty()) {
        throw std::invalid_argument("Tensor shape cannot be empty.");
    }
    for (auto dim : shape_.dims) {
        if (dim <= 0) {
            throw std::invalid_argument("Tensor dimensions must be positive.");
        }
    }

    // Compute strides
    compute_strides();

    // Allocate memory
    allocate_memory();
}

