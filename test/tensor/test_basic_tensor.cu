#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cassert>
#include "../../include/tensor/basic_tensor.h"
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

extern "C" bool test_basic_tensor_copy_constructor() {
    std::cout << "Testing BasicTensor Copy Constructor" << std::endl;
    
    try {
        std::vector<size_t> shape = {3, 4};
        float* data = new float[12];
        // Initialize with test data
        for (size_t i = 0; i < 12; i++) {
            data[i] = static_cast<float>(i);
        }
        BasicTensor<float> tensor(2, shape, data);
        delete[] data; // free the data after creating the tensor
        BasicTensor<float> tensor_copy(tensor);
        
        bool success = true;
        success &= tensor_copy.n_dims == 2;
        success &= tensor_copy.total_size == 12;
        success &= tensor_copy.get_element(0) == 0;
        success &= tensor_copy.get_element(11) == 11;
        
        std::cout << "  - Tensor copy created successfully" << std::endl;
        std::cout << "  - n_dims: " << tensor_copy.n_dims << std::endl;
        std::cout << "  - total_size: " << tensor_copy.total_size << std::endl;
        std::cout << "  - First element: " << tensor_copy.get_element(0) << std::endl;
        std::cout << "  - Last element: " << tensor_copy.get_element(11) << std::endl;
        
        return success;
    }
    catch (const std::exception& e) {
        std::cerr << "!!! Test failed: " << e.what() << std::endl;
        return false;
    }
}

extern "C" bool test_basic_tensor_to_device_data(){
    std::cout << "Testing BasicTensor to Device Data" << std::endl;
    try {
        std::vector<size_t> shape = {3, 4};
        float data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        BasicTensor<float> tensor(2, shape, data);
        
        tensor.set_element(0, 10);
        bool success = true;
        success &= tensor.get_element(0) == 10;
        if (success) {
            std::cout << "  - Tensor to device data test passed" << std::endl;
        } else {
            std::cout << "  - Tensor to device data test failed" << std::endl;
        }

        return success;
    }
    catch (const std::exception& e) {
        std::cerr << "!!! Test failed: " << e.what() << std::endl;
        return false;
    }
}


extern "C" bool test_basic_tensor_get_element() {
    std::cout << "Testing BasicTensor Get Element" << std::endl;
    try {
        std::vector<size_t> shape = {3, 4};
        float data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        BasicTensor<float> tensor(2, shape, data);

        bool success = true;
        success &= tensor.get_element(0) == 1;
        success &= tensor.get_element(11) == 12;

        if (success) {
            std::cout << "  - Tensor get element test passed" << std::endl;
        } else {
            std::cout << "  - Tensor get element test failed" << std::endl;
        }
        return success;
    }
    catch (const std::exception& e) {
        std::cerr << "!!! Test failed: " << e.what() << std::endl;
        return false;
    }
}



extern "C" bool test_memory_leak_tensor() {
    std::cout << "Testing Memory Leak in Tensor" << std::endl;
    try {
        std::vector<size_t> shape = {3, 4};
        float data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        
        float* data_address = data;
        BasicTensor<float> tensor(2, shape, data);

        bool success = true;
        success &= tensor.n_dims == 2;
        success &= tensor.total_size == 12;
        success &= tensor.get_element(0) == 1;
        success &= tensor.get_element(11) == 12;
        if (success) {
            std::cout << "  - Tensor created successfully" << std::endl;
        } else {
            std::cout << "  - Tensor creation failed" << std::endl;
        }

        success &= data_address != tensor.host_data;
        if (!success) {
            std::cout << "  - data_address: " << data_address << std::endl;
            std::cout << "  - tensor.host_data: " << tensor.host_data << std::endl;
        }

        // check if the data is copied to the tensor
        success &= data_address != tensor.host_data;
        if (!success) {
            std::cout << "  - data_address: " << data_address << std::endl;
            std::cout << "  - tensor.host_data: " << tensor.host_data << std::endl;
        }

        return success;
    }
    catch (const std::exception& e) {
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

extern "C" bool test_basic_tensor_at() {
    try {
        std::vector<size_t> shape = {2, 3};
        float data[] = {1, 2, 3, 4, 5, 6};
        BasicTensor<float> tensor(2, shape, data);
        
        bool success = true;
        success &= tensor.at({0, 0}) == 1;
        success &= tensor.at({1, 2}) == 6;
        
        std::cout << "At test completed!" << std::endl;
        return success;
    }
    catch (const std::exception& e) {
        std::cerr << "At test error: " << e.what() << std::endl;
        return false;
    }
}

extern "C" bool test_basic_tensor_set_at() {
    try {
        std::vector<size_t> shape = {2, 3};
        float data[] = {1, 2, 3, 4, 5, 6};
        BasicTensor<float> tensor(2, shape, data);
        
        bool success = true;
        tensor.set_at({0, 0}, 10);
        success &= tensor.get_element(0) == 10;
        
        std::cout << "Set at test completed!" << std::endl;
        return success;
    }
    catch (const std::exception& e) {
        std::cerr << "Set at test error: " << e.what() << std::endl;
        return false;
    }
}


extern "C" bool test_basic_tensor_contraction() {
    try {
        std::vector<size_t> shape1= {2, 3};
        std::vector<size_t> shape2 = {3, 4};
        float data1[] = {1, 2, 3, 4, 5, 6}; // {{1, 2, 3}, {4, 5, 6}}
        float data2[] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120}; // {{10, 20, 30, 40}, {50, 60, 70, 80}, {90, 100, 110, 120}}
        BasicTensor<float> tensor1(2, shape1, data1);
        BasicTensor<float> tensor2(2, shape2, data2);


        /**
         * 1 2 3  *  10 20 30 40     =  (1*10 + 2*50 + 3*90) (1*20 + 2*60 + 3*100) (1*30 + 2*70 + 3*110) (1*40 + 2*80 + 3*120)
         * 4 5 6  *  50 60 70 80
         *          90 100 110 120
         * */       
        std::vector<size_t[2]> contraction_indices = {{0, 1}};
        BasicTensor<float> result = tensor1.contraction(tensor2, contraction_indices);

        std::vector<size_t> result_shape = result.get_shape(); // {2, 4}
        bool success = true;
        success &= result_shape.size() == 2;
        success &= result_shape[0] == 2;
        success &= result_shape[1] == 4;
        success &= result.get_element(0) == 140; // 1*10 + 2*50 + 3*90
        success &= result.get_element(1) == 320; // 1*20 + 2*60 + 3*100
    
        return success;
    }
    catch (const std::exception& e) {
        std::cerr << "Contraction test error: " << e.what() << std::endl;
        return false;
    }
}

// TODO: Add test for the BasicTensorDevice class
