#include <iostream>
#include <cassert>
#include <cuda_runtime.h>

// Simple testing macros
#define ASSERT_EQ(a, b, msg) \
    if ((a) != (b)) { \
        std::cerr << "❌ FAIL: " << msg << " - Expected: " << (b) << ", Got: " << (a) << std::endl; \
        return false; \
    } else { \
        std::cout << "✅ PASS: " << msg << std::endl; \
    }

#define ASSERT_FLOAT_EQ(a, b, tolerance, msg) \
    if (fabs((a) - (b)) > (tolerance)) { \
        std::cerr << "❌ FAIL: " << msg << " - Expected: " << (b) << ", Got: " << (a) << std::endl; \
        return false; \
    } else { \
        std::cout << "✅ PASS: " << msg << std::endl; \
    }

// Simple CUDA kernel for testing
__global__ void simpleAdd(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Test function
extern "C" bool test_simple_addition() {
    std::cout << "\n🧪 Testing Simple CUDA Addition..." << std::endl;
    
    const int N = 5;
    float h_a[] = {1, 2, 3, 4, 5};
    float h_b[] = {2, 3, 4, 5, 6};
    float h_c[N];
    
    // GPU arrays
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));
    
    // Copy to GPU
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel
    simpleAdd<<<1, N>>>(d_a, d_b, d_c, N);
    
    // Copy result back
    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify results
    ASSERT_FLOAT_EQ(h_c[0], 3.0f, 1e-6, "First element should be 3");
    ASSERT_FLOAT_EQ(h_c[1], 5.0f, 1e-6, "Second element should be 5");
    ASSERT_FLOAT_EQ(h_c[4], 11.0f, 1e-6, "Last element should be 11");
    
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return true;
}

extern "C" bool test_basic_operations() {
    std::cout << "\n🧪 Testing Basic Operations..." << std::endl;
    
    int a = 5, b = 10;
    ASSERT_EQ(a + b, 15, "Addition should work");
    ASSERT_EQ(a * b, 50, "Multiplication should work");
    
    return true;
}

int main() {
    std::cout << "🚀 Simple CUDA Testing Demo" << std::endl;
    
    bool success = true;
    success &= test_basic_operations();
    success &= test_simple_addition();
    
    if (success) {
        std::cout << "\n🎉 All tests passed!" << std::endl;
    } else {
        std::cout << "\n💥 Some tests failed!" << std::endl;
    }
    
    return success ? 0 : 1;
}
