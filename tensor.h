#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cstring>
#include <ostream>

// =============================
// Data Type Definitions
// =============================

// Data types supported by the tensor
enum class Dtype {
    Int16, Int32, Int64,
    Bfloat16, Float16,
    Float32, Float64
};

// Compute element size for a given Dtype
inline size_t dtype_size(Dtype dtype) {
    switch (dtype) {
        case Dtype::Int16:    return 2;
        case Dtype::Int32:    return 4;
        case Dtype::Int64:    return 8;
        case Dtype::Bfloat16: return 2;
        case Dtype::Float16:  return 2;
        case Dtype::Float32:  return 4;
        case Dtype::Float64:  return 8;
        default: throw std::invalid_argument("Unsupported Dtype");
    }
}

// =============================
// Device Management
// =============================

enum class DeviceType {
    CPU,
    CUDA // Future GPU support
};

struct Device {
    DeviceType type;
    int index; // For multi-GPU systems (default 0)

    Device(DeviceType type_ = DeviceType::CPU, int index_ = 0)
        : type(type_), index(index_) {}
};

// =============================
// Shape and Stride Structures
// =============================

struct Shape {
    std::vector<int32_t> dims;

    Shape() = default;
    explicit Shape(const std::vector<int32_t>& d) : dims(d) {}
};

struct Stride {
    std::vector<int32_t> strides;

    Stride() = default;
    explicit Stride(const std::vector<int32_t>& s) : strides(s) {}
};

// =============================
// Tensor Class
// =============================
class Tensor {
public:
    // =========================
    // Constructors
    // =========================

    // Primary constructor
    Tensor(const Shape& shape, Dtype dtype, const Device& device = Device(),
           bool requires_grad = false);

    // Default constructor
    Tensor() = default;

    // Copy constructor (shares underlying data)
    Tensor(const Tensor& other) = default;

    // Move constructor
    Tensor(Tensor&& other) noexcept = default;

    // Assignment operators
    Tensor& operator=(const Tensor& other) = default;
    Tensor& operator=(Tensor&& other) noexcept = default;

    // =========================
    // Metadata Accessors
    // =========================
    const std::vector<int32_t>& shape() const { return shape_.dims; }
    const std::vector<int32_t>& stride() const { return stride_.strides; }
    Dtype dtype() const { return dtype_; }
    const Device& device() const { return device_; }
    bool requires_grad() const { return requires_grad_; }
    bool is_owner() const { return is_owner_; }

    // Total number of elements in the tensor
    size_t numel() const;

    // Total memory size in bytes
    size_t nbytes() const { return numel() * dtype_size(dtype_); }

    // Raw data access
    template <typename T>
    T* data() {
        return reinterpret_cast<T*>(data_.get());
    }

    template <typename T>
    const T* data() const {
        return reinterpret_cast<const T*>(data_.get());
    }

private:
    // =========================
    // Helper Methods
    // =========================
    void compute_strides(); // Computes strides from shape
    void allocate_memory(); // Allocates memory based on device

private:
    // =========================
    // Member Variables
    // =========================
    Shape shape_;
    Stride stride_;
    Dtype dtype_;
    Device device_;
    bool requires_grad_ = false;
    bool is_owner_ = false; // Tracks if tensor owns its memory

    std::shared_ptr<uint8_t> data_; // Smart pointer for data
};

