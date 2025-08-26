#include "include/tensor/basic_tensor.h"

/**
 * @brief Device kernel for tensor addition using BasicTensorDevice
 * @param tensor1 First tensor device structure
 * @param tensor2 Second tensor device structure  
 * @param result Result tensor device structure
 */
template<typename T>
__global__ void add_tensor_device_kernel(BasicTensorDevice<T> tensor1, BasicTensorDevice<T> tensor2, BasicTensorDevice<T> result) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < tensor1.total_size) {
        result.data[idx] = tensor1.data[idx] + tensor2.data[idx];
    }
}

template<typename T>
__global__ void contraction_device_kernel(BasicTensorDevice<T> tensor1, BasicTensorDevice<T> tensor2, BasicTensorDevice<T> result, int[][2] contraction_indices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < tensor1.total_size && idx2 < tensor2.total_size) {
        result.data[idx][idx2] = tensor1.data[idx][idx2] * tensor2.data[idx2][idx];
    }
}



/**
 * @brief Constructor for BasicTensor
 * @param n_dims The number of dimensions of the tensor
 * @param shape The shape of the tensor
 * @param host_data The data of the tensor (will be copied to device and host)
 */
template<typename T>
__host__ BasicTensor<T>::BasicTensor(int n_dims, std::vector<size_t> shape, T* host_data) : n_dims(n_dims) {
    if (n_dims <= 0) {
        throw std::invalid_argument("Number of dimensions must be greater than 0");
    }

    // check if shape is valid
    if (shape.empty()) {
        throw std::invalid_argument("Shape must be provided");
    }

    if (n_dims != shape.size()) {
        throw std::invalid_argument("Number of dimensions should match with the size of shape array!");
    }

    // calculate total size
    total_size = 1;
    for (int i = 0; i < n_dims; i++) {
        total_size *= shape[i];
    }

    this->shape = std::vector<size_t>(shape);

    if (host_data != nullptr) {
        this->host_data = new T[total_size];
        for (int i = 0; i < total_size; i++) {
            this->host_data[i] = host_data[i];
        }
    } else {
        this->host_data = nullptr;
    }
}

/**
 * @brief Copy constructor
 */
template<typename T>
__host__ BasicTensor<T>::BasicTensor(const BasicTensor& other) {
    std::cout << "Copy constructor called" << std::endl;

    // Copy basic properties
    n_dims = other.n_dims;
    shape = std::vector<size_t>(other.shape);
    total_size = other.total_size;

    // Copy host data if it exists
    if (other.host_data != nullptr) {
        host_data = new T[total_size];
        for (int i = 0; i < total_size; i++) {
            host_data[i] = other.host_data[i];
        }
    } else {
        host_data = nullptr;
    }

}

/**
 * @brief Destructor for BasicTensor
 */
template<typename T>
__host__ BasicTensor<T>::~BasicTensor() {
    if (host_data != nullptr) {
        delete[] host_data;
    }
}

/**
 * @brief Get element at the given index (copies from device to host)
 * @param idx The index of the element to access
 * @return The value at the given index
 */
template<typename T>
__host__ T BasicTensor<T>::get_element(int idx) const {
    if (idx < 0 || idx >= total_size) {
        throw std::out_of_range("Index out of range");
    }

    if (host_data == nullptr) {
        throw std::runtime_error("Host data is not allocated");
    }

    return host_data[idx];
}

/**
 * @brief Set element at the given index (copies from host to device)
 * @param idx The index of the element to set
 * @param value The value to set
 */
template<typename T>
__host__ void BasicTensor<T>::set_element(int idx, T value)  {
    if (idx < 0 || idx >= total_size) {
        throw std::out_of_range("Index out of range");
    }

    if (host_data == nullptr) {
        throw std::runtime_error("Host data is not allocated");
    }

    host_data[idx] = value;
}

/**
 * @brief Copy device data to internal host array
 * @param device_data Device array to copy data from
 */
template<typename T>
__host__ void BasicTensor<T>::copy_from_device_to_host_data(T* device_data) {
    if (this->host_data == nullptr) {
        this->host_data = new T[total_size];
    }
    cudaMemcpy(this->host_data, device_data, total_size * sizeof(T), cudaMemcpyDeviceToHost);
}

/**
 * @brief Get the shape of the tensor
 * @return A pointer to the shape of the tensor
 */
template<typename T>
__host__ std::vector<size_t> BasicTensor<T>::get_shape() const {
    return shape;
}

/**
 * @brief Get the total size of the tensor
 * @return The total size of the tensor
 */
template<typename T>
__host__ int BasicTensor<T>::get_total_size() const {
    return total_size;
}

/**
 * @brief Compare two tensor dimensions
 * @param other The other tensor to compare to
 * @return True if the dimensions are the same, false otherwise
 */
template<typename T>
__host__ bool BasicTensor<T>::compare_dimensions(const BasicTensor<T>& other) const {
    return n_dims == other.n_dims && total_size == other.total_size && shape == other.shape;
}

/**
 * @brief Compare two tensor dimensions for contraction
 * @param other The other tensor to compare to
 * @param contraction_indices The indices of the contraction
 * @return True if the dimensions are compatible for contraction, false otherwise
 */
template<typename T>
__host__ bool BasicTensor<T>::compare_contraction_dimensions(const BasicTensor<T>& other, const std::vector<size_t[2]>& contraction_indices) const {
    for (auto& contr : contraction_indices) {
        if (contr[0] >= this->n_dims || contr[0] >= other.n_dims || contr[1] >= other.n_dims || contr[1] >= this->n_dims || contr[0] < 0 || contr[1] < 0) {
            throw std::invalid_argument("Contraction indices must be within bounds [0, n_dims)");
        }
        if (contr[0] == contr[1]) {
            throw std::invalid_argument("Contraction indices must be unique and different");
        }
        if (this->shape[contr[0]] != other.shape[contr[1]]) {
            throw std::invalid_argument("Dimension sizes must match along contraction indices");
        }
    }

    return true;  // All checks passed
}

template<typename T>
__host__ size_t BasicTensor<T>::compute_flat_index(const std::vector<size_t>& indices) const {
    if (indices.size() != static_cast<size_t>(n_dims)) {
        throw std::invalid_argument("Indices size must match n_dims");
    }
    size_t idx = 0;
    if (n_dims > 0) {
        idx = indices[0];
        if (indices[0] >= shape[0]) {
            throw std::out_of_range("Index out of bounds");
        }
        for (int d = 1; d < n_dims; ++d) {
            if (indices[d] >= shape[d]) {
                throw std::out_of_range("Index out of bounds");
            }
            idx = idx * shape[d] + indices[d];
        }
    }
    return idx;
}

/**
 * @brief Get element at the given indices
 * @param indices The indices of the element to access
 * @return The value at the given indices
 */
template<typename T>
__host__ T BasicTensor<T>::at(const std::vector<size_t>& indices) const {
    size_t idx = compute_flat_index(indices);
    return host_data[idx];
}

template<typename T>
__host__ void BasicTensor<T>::set_at(const std::vector<size_t>& indices, T value) {
    size_t idx = compute_flat_index(indices);
    host_data[idx] = value;
}

/**
 * @brief Assignment operator
 * TODO: Add better explanation
 */
template<typename T>
__host__ BasicTensor<T>& BasicTensor<T>::operator=(const BasicTensor<T>& other) {
    if (this != &other) {
        // Clean up existing resources
        if (host_data != nullptr) {
            delete[] host_data;
        }
        
        // Copy new values
        n_dims = other.n_dims;
        shape = other.shape;
        total_size = other.total_size;
        
        // Copy host data if it exists
        if (other.host_data != nullptr) {
            host_data = new T[total_size];
            for (int i = 0; i < total_size; i++) {
                host_data[i] = other.host_data[i];
            }
        } else {
            host_data = nullptr;
        }
    }
    return *this;
}

/**
 * @brief Add two tensors  
 * @param other The other tensor to add
 * @return A new tensor that is the sum of the two tensors
 */
template<typename T>
__host__ BasicTensor<T> BasicTensor<T>::operator+(const BasicTensor<T>& other) const {
    if (!compare_dimensions(other)) {
        throw std::invalid_argument("Tensors must have the same dimensions");
    }

    // Create empty result tensor
    BasicTensor<T> result(this->n_dims, this->shape, nullptr);

    // Convert to device structures
    BasicTensorDevice<T> dev_this = this->to_device();
    BasicTensorDevice<T> dev_other = other.to_device();
    BasicTensorDevice<T> dev_result = result.to_device();

    // Calculate grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_size + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the device kernel
    add_tensor_device_kernel<<<blocksPerGrid, threadsPerBlock>>>(dev_this, dev_other, dev_result);
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Clean up on error
        cudaFree(dev_this.shape);
        cudaFree(dev_this.data);
        cudaFree(dev_other.shape);
        cudaFree(dev_other.data);
        cudaFree(dev_result.shape);
        cudaFree(dev_result.data);
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
    
    // Wait for kernel to complete and check for execution errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        // Clean up on error
        cudaFree(dev_this.shape);
        cudaFree(dev_this.data);
        cudaFree(dev_other.shape);
        cudaFree(dev_other.data);
        cudaFree(dev_result.shape);
        cudaFree(dev_result.data);
        throw std::runtime_error("CUDA kernel execution failed: " + std::string(cudaGetErrorString(err)));
    }
    
    // Copy result data back to host
    result.copy_from_device_to_host_data(dev_result.data);
    
    // Clean up device memory allocated by to_device()
    cudaFree(dev_this.shape);
    cudaFree(dev_this.data);
    cudaFree(dev_other.shape);
    cudaFree(dev_other.data);
    cudaFree(dev_result.shape);
    cudaFree(dev_result.data);

    return result;
}

template<typename T>
__host__ BasicTensor<T> BasicTensor<T>::contraction(const BasicTensor<T>& other, const std::vector<size_t[2]>& contraction_indices) const {
    if (!compare_contraction_dimensions(other, contraction_indices)) {
        throw std::invalid_argument("Tensors must have compatible contraction dimensions for contraction");
    }

    // Create empty result tensor
    BasicTensor<T> result(this->n_dims, this->shape, nullptr);

    // Convert to device structures
    BasicTensorDevice<T> dev_this = this->to_device();
    BasicTensorDevice<T> dev_other = other.to_device();
    BasicTensorDevice<T> dev_result = result.to_device();

    // Calculate grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_size + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the device kernel
    contraction_device_kernel<<<blocksPerGrid, threadsPerBlock>>>(dev_this, dev_other, dev_result, contraction_indices);
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(dev_this.shape);
        cudaFree(dev_this.data);
        cudaFree(dev_other.shape);
        cudaFree(dev_other.data);
        cudaFree(dev_result.shape);
        cudaFree(dev_result.data);
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }

    // Wait for kernel to complete and check for execution errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(dev_this.shape);
        cudaFree(dev_this.data);
        cudaFree(dev_other.shape);
        cudaFree(dev_other.data);
        cudaFree(dev_result.shape);
        cudaFree(dev_result.data);
        throw std::runtime_error("CUDA kernel execution failed: " + std::string(cudaGetErrorString(err)));
    }

    // Copy result data back to host
    result.copy_from_device_to_host_data(dev_result.data);

    // Clean up device memory allocated by to_device()
    cudaFree(dev_this.shape);
    cudaFree(dev_this.data);
    cudaFree(dev_other.shape);
    cudaFree(dev_other.data);
    cudaFree(dev_result.shape);
    cudaFree(dev_result.data);

    return result;
}

/**
 * @brief Convert to device-compatible structure
 * @return BasicTensorDevice structure that can be used in kernels
 */
template<typename T>
__host__ BasicTensorDevice<T> BasicTensor<T>::to_device() const {
    // Allocate device memory for shape
    size_t* d_shape;
    cudaMalloc(&d_shape, n_dims * sizeof(size_t));
    cudaMemcpy(d_shape, shape.data(), n_dims * sizeof(size_t), cudaMemcpyHostToDevice);
    
    // Allocate device memory for data and copy from host
    T* d_data;
    cudaMalloc(&d_data, total_size * sizeof(T));
    if (host_data != nullptr) {
        cudaMemcpy(d_data, host_data, total_size * sizeof(T), cudaMemcpyHostToDevice);
    } else {
        cudaMemset(d_data, 0, total_size * sizeof(T));
    }
    
    return BasicTensorDevice<T>(n_dims, d_shape, total_size, d_data);
}

/**
 * @brief Device constructor - can be called from both host and device
 */
template<typename T>
__device__ __host__ BasicTensorDevice<T>::BasicTensorDevice(int n_dims, size_t* shape, int total_size, T* data)
    : n_dims(n_dims), shape(shape), total_size(total_size), data(data) {}

/**
 * @brief Device copy constructor - can be called from both host and device
 * Applies copy construction (Copies attributes of the same object type to current object type)
 */
template<typename T>
__device__ __host__ BasicTensorDevice<T>::BasicTensorDevice(const BasicTensorDevice<T>& other) 
    : n_dims(other.n_dims), shape(other.shape), total_size(other.total_size), data(other.data) {}

/**
 * @brief Device destructor - can be called from both host and device
 * Note: BasicTensorDevice doesn't own the memory, so no cleanup needed
 */
template<typename T>
__device__ __host__ BasicTensorDevice<T>::~BasicTensorDevice() {
    // No cleanup needed - BasicTensorDevice is just a view of device memory
    // Memory management is handled by the calling code such as operator+
}

template<typename T>
__device__ size_t BasicTensorDevice<T>::compute_flat_index(const size_t* indices) const {
    size_t idx = 0;
    if (n_dims > 0) {
        idx = indices[0];
        for (int d = 1; d < n_dims; ++d) {
            idx = idx * shape[d] + indices[d];
        }
    }
    return idx;
}

template<typename T>
__device__ T& BasicTensorDevice<T>::at(const size_t* indices) {
    return data[compute_flat_index(indices)];
}

template<typename T>
__device__ const T& BasicTensorDevice<T>::at(const size_t* indices) const {
    return data[compute_flat_index(indices)];
}

/**
 * @brief Device-side element access
 */
template<typename T>
__device__ T& BasicTensorDevice<T>::operator[](int idx) {
    return data[idx];
}

/**
 * @brief Device-side const element access
 */
template<typename T>
__device__ const T& BasicTensorDevice<T>::operator[](int idx) const {
    return data[idx];
}

//Test manually the BasicTensor class
#ifndef SKIP_MAIN
int main() {
    try {
        try {
            std::vector<size_t> shape = {3, 4, 5};
            float* data = new float[12];
            BasicTensor<float> tensor(2, shape, data);
        } catch (const std::invalid_argument& e) {
            std::cout << "Test passed: " << e.what() << std::endl;
            return 1;
        }


        std::vector<size_t> shape = {2, 3};
        float data1[] = {1, 2, 3, 4, 5, 6};
        float data2[] = {10, 20, 30, 40, 50, 60};
        
        BasicTensor<float> tensor1(2, shape, data1);
        BasicTensor<float> tensor2(2, shape, data2);
        
        std::cout << "Testing tensor addition..." << std::endl;
        BasicTensor<float> result = tensor1 + tensor2;
        
        // Copy result back to host to verify
        float host_result[6];
        for (int i = 0; i < 6; i++) {
            host_result[i] = result.get_element(i);
        }
        
        std::cout << "Result: ";
        for (int i = 0; i < 6; i++) {
            std::cout << host_result[i] << " ";
        }
        std::cout << std::endl;
        
        std::cout << "Test completed successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
#endif // SKIP_MAIN


/**
 * @brief Test instantiations of the BasicTensor class
 * This is used to test the instantiations of the BasicTensor class
 * Otherwise, compilation will fail because of the template instantiations
 */
#ifdef TEST_INSTANTIATIONS
// Minimal instantiations for essential types only
template class BasicTensor<float>;
template class BasicTensor<int>;
template class BasicTensor<double>;
template class BasicTensorDevice<float>;
template class BasicTensorDevice<int>;
template class BasicTensorDevice<double>;

// Kernel instantiations
template __global__ void add_tensor_device_kernel<float>(BasicTensorDevice<float>, BasicTensorDevice<float>, BasicTensorDevice<float>);
template __global__ void add_tensor_device_kernel<int>(BasicTensorDevice<int>, BasicTensorDevice<int>, BasicTensorDevice<int>);
template __global__ void add_tensor_device_kernel<double>(BasicTensorDevice<double>, BasicTensorDevice<double>, BasicTensorDevice<double>);
#endif // TEST_INSTANTIATIONS