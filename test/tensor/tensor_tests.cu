#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cassert>
#include "../../include/tensor/basic_tensor.h"

// Include template implementations for instantiation
#define BASIC_TENSOR_SKIP_MAIN
#include "../../src/tensor/basic_tensor.cu"

// CUDA testing utilities
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

extern "C" bool test_basic_tensor_constructur() {
    std::cout << "Testing BasicTensor Constructor" << std::endl;
    
    try {
        // Create a simple 2D tensor (3x4 matrix)
        std::vector<size_t> shape = {3, 4};
        float* data = new float[12];  // 3*4 = 12 elements
        
        // Initialize with test data
        for (size_t i = 0; i < 12; i++) {
            data[i] = static_cast<float>(i);
        }
        
        BasicTensor<float> tensor(2, shape, data);

        bool success = true;

        success &= tensor.n_dims == 2;
        success &= tensor.total_size == 12;
        success &= tensor.get_element(0) == 0;
        success &= tensor.get_element(11) == 11;
        
        // Test basic properties
        std::cout << "  - Tensor created successfully" << std::endl;
        std::cout << "  - n_dims: " << tensor.n_dims << std::endl;
        std::cout << "  - total_size: " << tensor.total_size << std::endl;
        std::cout << "  - First element: " << tensor.get_element(0) << std::endl;
        std::cout << "  - Last element: " << tensor.get_element(11) << std::endl;
        
        return success;
        
    } catch (const std::exception& e) {
        std::cerr << "!!! Test failed: " << e.what() << std::endl;
        return false;
    }
}

extern "C" bool test_basic_tensor_constructor_invalid_shape() {
    std::cout << "Testing BasicTensor Constructor with invalid shape" << std::endl;
    
    try {
        std::vector<size_t> shape = {3, 4, 5};
        float* data = new float[12];
        BasicTensor<float> tensor(2, shape, data);
        return false;
    } catch (const std::invalid_argument& e) {
        return true;
    }catch (const std::exception& e) {
        return true;
    } 
}


extern "C" bool test_basic_tensor_add() {
    try {
        std::vector<size_t> shape = {2, 3};
        float data1[] = {1, 2, 3, 4, 5, 6};
        float data2[] = {10, 20, 30, 40, 50, 60};
        
        BasicTensor<float> tensor1(2, shape, data1);
        BasicTensor<float> tensor2(2, shape, data2);
        
        std::cout << "Testing tensor addition..." << std::endl;
        BasicTensor<float> result = tensor1 + tensor2;
        
        bool success = true;
        success &= result.n_dims == 2;
        success &= result.total_size == 6;
        // Cannot directly compare vectors with ==, need to compare elements
        std::vector<size_t> result_shape = result.get_shape();
        success &= result_shape.size() == shape.size();
        for (size_t i = 0; i < shape.size(); i++) {
            success &= result_shape[i] == shape[i];
        }
        
        success &= result.get_element(0) == 11;
        success &= result.get_element(1) == 22;
        success &= result.get_element(2) == 33;

        std::cout << "  - Result tensor: " << result.get_element(0) << ", " << result.get_element(1) << ", " << result.get_element(2) << std::endl;
        
        std::cout << "Test completed successfully!" << std::endl;
        return success;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return false;
    }
}

extern "C" bool test_basic_tensor_int() {
    try {
        std::vector<size_t> shape = {2, 3};
        int data1[] = {1, 2, 3, 4, 5, 6};
        int data2[] = {10, 20, 30, 40, 50, 60};
        
        BasicTensor<int> tensor1(2, shape, data1);
        BasicTensor<int> tensor2(2, shape, data2);
        
        std::cout << "Testing integer tensor addition..." << std::endl;
        BasicTensor<int> result = tensor1 + tensor2;
        
        bool success = true;
        success &= result.get_element(0) == 11;
        success &= result.get_element(1) == 22;
        success &= result.get_element(2) == 33;
        
        std::cout << "Integer test completed!" << std::endl;
        return success;
    }
    catch (const std::exception& e) {
        std::cerr << "Integer test error: " << e.what() << std::endl;
        return false;
    }
}

extern "C" bool test_basic_tensor_double() {
    try {
        std::vector<size_t> shape = {2, 2};
        double data1[] = {1.5, 2.5, 3.5, 4.5};
        double data2[] = {0.5, 0.5, 0.5, 0.5};
        
        BasicTensor<double> tensor1(2, shape, data1);
        BasicTensor<double> tensor2(2, shape, data2);
        
        std::cout << "Testing double tensor addition..." << std::endl;
        BasicTensor<double> result = tensor1 + tensor2;
        
        bool success = true;
        success &= abs(result.get_element(0) - 2.0) < 1e-10;
        success &= abs(result.get_element(1) - 3.0) < 1e-10;
        
        std::cout << "Double test completed!" << std::endl;
        return success;
    }
    catch (const std::exception& e) {
        std::cerr << "Double test error: " << e.what() << std::endl;
        return false;
    }
}

// Explicit template instantiations
template class BasicTensor<float>;
template class BasicTensor<int>;
template class BasicTensor<double>;
template class BasicTensorDevice<float>;
template class BasicTensorDevice<int>;
template class BasicTensorDevice<double>;
