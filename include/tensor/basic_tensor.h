#ifndef BASIC_TENSOR_H
#define BASIC_TENSOR_H
#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>
#include <vector>
// Forward declaration
template<typename T>
struct BasicTensorDevice;

template<typename T>
struct BasicTensor {
    int n_dims; // number of dimensions
    std::vector<size_t> shape; // shape of the tensor [3, 2, 5] means 3 legs 
    int total_size; // total size of the tensor
    T* host_data; // host data of the tensor

    __host__ BasicTensor(int n_dims, std::vector<size_t> shape, T* host_data);
    
    __host__ BasicTensor(const BasicTensor& other);
    
    __host__ ~BasicTensor();
    
    __host__ T get_element(int idx) const;

    
    __host__ void set_element(int idx, T value);

    __host__ void copy_from_device_to_host_data(T* device_data);
    
    __host__ std::vector<size_t> get_shape() const;

    
    __host__ int get_total_size() const;

    
    __host__ bool compare_dimensions(const BasicTensor<T>& other) const;

    __host__ bool compare_contraction_dimensions(const BasicTensor<T>& other, const std::vector<size_t[2]>& contraction_indices) const;

    __host__ size_t compute_flat_index(const std::vector<size_t>& indices) const;

    __host__ T at(const std::vector<size_t>& indices) const;

    __host__ void set_at(const std::vector<size_t>& indices, T value);

    __host__ BasicTensor<T>& operator=(const BasicTensor<T>& other);

    __host__ BasicTensor<T> operator+(const BasicTensor<T>& other) const;

    __host__ BasicTensor<T> contraction(const BasicTensor<T>& other, const std::vector<size_t[2]>& contraction_indices) const;

    __host__ BasicTensorDevice<T> to_device() const;
};

/**
 * @brief Device-compatible tensor structure for GPU operations
 * This struct can be used in CUDA kernels and contains only the essential data
 * @param n_dims The number of dimensions of the tensor
 * @param shape The shape of the tensor
 * @param total_size The total size of the tensor
 * @param data The data of the tensor (device memory)
 */
template<typename T>
struct BasicTensorDevice {
    int n_dims;
    size_t* shape;
    int total_size;
    T* data;

    __device__ __host__ BasicTensorDevice(int n_dims, size_t* shape, int total_size, T* data);

    __device__ __host__ BasicTensorDevice(const BasicTensorDevice<T>& other);

    __device__ __host__ ~BasicTensorDevice();

    __device__ __host__ size_t compute_flat_index(const size_t* indices) const;

    __device__ __host__ T& at(const size_t* indices);

    __device__ __host__ const T& at(const size_t* indices) const;

    __device__ T& operator[](int idx);

    __device__ const T& operator[](int idx) const;
};

#endif // BASIC_TENSOR_H