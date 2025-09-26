#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <memory>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <type_traits>

// =======================================
// Device Types
// =======================================
enum class DeviceType {
    CPU,
    CUDA // Future support
};

struct Device {
    DeviceType type;
    explicit Device(DeviceType t = DeviceType::CPU) : type(t) {}
};

// =======================================
// Data Types
// =======================================
enum class Dtype {
    Float32,
    Int32
};

// =======================================
// Shape (Wrapper for dimensions)
// =======================================
struct Shape {
    std::vector<size_t> dims;

    Shape() = default;
    Shape(std::initializer_list<size_t> dimensions) : dims(dimensions) {}
};

// =======================================
// Utilities: recursive shape inference and flattening
// =======================================

// Recursive shape inference for any nested vector or scalar
template <typename T>
Shape infer_shape(const T&) {
    return Shape{}; // Base case: scalar has empty dims
}

// Specialization for std::vector
template <typename T>
Shape infer_shape(const std::vector<T>& vec) {
    if (vec.empty()) return Shape{0};
    Shape s = infer_shape(vec[0]); // recursive call
    Shape result;
    result.dims.push_back(vec.size()); // current dimension
    result.dims.insert(result.dims.end(), s.dims.begin(), s.dims.end());
    return result;
}

// Helper to get innermost type
template <typename T>
struct innermost_type {
    using type = T;
};

template <typename T>
struct innermost_type<std::vector<T>> {
    using type = typename innermost_type<T>::type;
};

// Recursive flatten
template <typename T, typename OutT = typename innermost_type<T>::type>
void flatten_recursive(const T& value, std::vector<OutT>& out) {
    if constexpr (std::is_same_v<T, OutT>) {
        out.push_back(value);  // scalar
    } else {
        for (const auto& v : value) {
            flatten_recursive(v, out); // recursive
        }
    }
}

// Wrapper
template <typename T>
auto flatten(const T& vec) {
    using OutT = typename innermost_type<T>::type;
    std::vector<OutT> out;
    flatten_recursive(vec, out);
    return out;
}

// =======================================
// Tensor Class
// =======================================
class Tensor {
private:
    Dtype dtype_ = Dtype::Float32;
    Device device_ = Device(DeviceType::CPU);
    std::vector<size_t> strides_;
    bool is_owner_ = true;

    // Helper: compute total number of elements
    size_t compute_numel() const {
        if (shape_.dims.empty()) return 0;
        return std::accumulate(shape_.dims.begin(), shape_.dims.end(),
                               static_cast<size_t>(1), std::multiplies<size_t>());
    }

    // Helper: compute strides for row-major layout
    void compute_strides() {
        if (shape_.dims.empty()) return;
        strides_.resize(shape_.dims.size());
        size_t stride_val = 1;
        for (int i = static_cast<int>(shape_.dims.size()) - 1; i >= 0; --i) {
            strides_[i] = stride_val;
            stride_val *= shape_.dims[i];
        }
    }

    // Helper: get element size in bytes
    size_t element_size() const {
        switch (dtype_) {
            case Dtype::Float32: return 4;
            case Dtype::Int32: return 4;
            default: throw std::runtime_error("Unsupported dtype");
        }
    }

    // Allocate raw memory
    void allocate_memory() {
        size_t total_bytes = compute_numel() * element_size();
        if (total_bytes == 0) {
            throw std::runtime_error("Cannot allocate zero-size tensor.");
        }

        if (device_.type == DeviceType::CPU) {
            data_ = std::shared_ptr<uint8_t>(new uint8_t[total_bytes],
                                             std::default_delete<uint8_t[]>());
        } else {
            throw std::runtime_error("CUDA allocation not implemented yet.");
        }

        is_owner_ = true;
    }

public:
    Shape shape_;
    std::shared_ptr<uint8_t> data_; // raw memory

    Tensor() = default; // Default constructor

    

    // ================================
    // Constructor: nested vector (auto-infer N-dim shape)
    // ================================
    template <typename T>
    Tensor(const std::vector<T>& vec, Dtype dtype,
           const Device& device, typename std::enable_if<std::is_same<T, std::vector<typename T::value_type>>::value>::type* = 0)
        : dtype_(dtype), device_(device)
    {
        shape_ = infer_shape(vec);
        allocate_memory();
        auto flat_data = flatten(vec);
        std::memcpy(data_.get(), flat_data.data(), flat_data.size() * sizeof(typename T::value_type));
    }

    // ================================
    // Constructor: flat vector + explicit shape
    // ================================
    template <typename T>
    Tensor(const std::vector<T>& vec, const Shape& shape,
           Dtype dtype = Dtype::Float32,
           const Device& device = Device(DeviceType::CPU))
        : shape_(shape), dtype_(dtype), device_(device)
    {
        size_t expected_size = 1;
        for (size_t dim : shape.dims) {
            if (dim <= 0)
                throw std::invalid_argument("Shape dimensions must be positive.");
            expected_size *= dim;
        }

        if (vec.size() != expected_size)
            throw std::invalid_argument("Data size does not match tensor shape.");

        allocate_memory();
        std::memcpy(data_.get(), vec.data(), vec.size() * sizeof(T));
    }

    // ================================
    // Constructor: allocate tensor only
    // ================================
    Tensor(const Shape& shape,
           Dtype dtype = Dtype::Float32,
           const Device& device = Device(DeviceType::CPU))
        : shape_(shape), dtype_(dtype), device_(device)
    {
        compute_strides();
        allocate_memory();
    }

    // Access raw data
    template <typename T>
    T* data() {
        return reinterpret_cast<T*>(data_.get());
    }

    // Total elements
    size_t numel() const { return compute_numel(); }

    // Debug: print tensor metadata
    void print_info() const {
        std::cout << "Tensor(";
        for (size_t i = 0; i < shape_.dims.size(); ++i) {
            std::cout << shape_.dims[i];
            if (i + 1 < shape_.dims.size()) std::cout << ",";
        }
        std::cout << ") dtype=" << (dtype_ == Dtype::Float32 ? "Float32" : "Int32") << std::endl;
    }

    // Print data
    template <typename T>
    void print_data() const {
        const T* ptr = reinterpret_cast<const T*>(data_.get());
        for (size_t i = 0; i < numel(); ++i) {
            std::cout << ptr[i] << " ";
        }
        std::cout << std::endl;
    }

    // Factory functions (declarations)
    static Tensor zeros(const Shape& shape, Dtype dtype = Dtype::Float32,const Device& device = Device(DeviceType::CPU));
    static Tensor ones(const Shape& shape, Dtype dtype = Dtype::Float32,const Device& device = Device(DeviceType::CPU));
};

#endif
