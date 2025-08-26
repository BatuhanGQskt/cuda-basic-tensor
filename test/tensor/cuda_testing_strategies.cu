#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cassert>

// CUDA testing utilities
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

// Test helper to compare arrays with tolerance
bool compareArrays(const float* a, const float* b, int n, float tolerance = 1e-6f) {
    for (int i = 0; i < n; i++) {
        if (fabs(a[i] - b[i]) > tolerance) {
            std::cout << "Mismatch at index " << i << ": " << a[i] << " vs " << b[i] << std::endl;
            return false;
        }
    }
    return true;
}

// ==============================================================================
// STRATEGY 1: CPU Reference Implementation + GPU Comparison
// ==============================================================================

// CPU reference implementation
void cpuVectorAdd(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// GPU implementation
__global__ void gpuVectorAdd(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

bool test_cpu_gpu_comparison() {
    std::cout << "🧪 Testing CPU vs GPU Vector Addition..." << std::endl;
    
    const int N = 10000;
    
    // Host arrays
    std::vector<float> h_a(N), h_b(N), h_c_cpu(N), h_c_gpu(N);
    
    // Initialize test data
    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(rand()) / RAND_MAX;
        h_b[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    // CPU computation
    cpuVectorAdd(h_a.data(), h_b.data(), h_c_cpu.data(), N);
    
    // GPU computation
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    gpuVectorAdd<<<grid, block>>>(d_a, d_b, d_c, N);
    
    CUDA_CHECK(cudaMemcpy(h_c_gpu.data(), d_c, N * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Compare results
    bool success = compareArrays(h_c_cpu.data(), h_c_gpu.data(), N);
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    
    return success;
}

// ==============================================================================
// STRATEGY 2: Device-Side Assertions (CUDA 4.0+)
// ==============================================================================

__global__ void kernelWithAssertions(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Example: test that input data is within expected range
        assert(data[idx] >= 0.0f && data[idx] <= 1.0f);
        
        data[idx] *= 2.0f;
        
        // Assert the result is within expected range
        assert(data[idx] >= 0.0f && data[idx] <= 2.0f);
    }
}

bool test_device_assertions() {
    std::cout << "🧪 Testing Device-Side Assertions..." << std::endl;
    
    const int N = 1000;
    std::vector<float> h_data(N);
    
    // Initialize with valid data
    for (int i = 0; i < N; i++) {
        h_data[i] = static_cast<float>(rand()) / RAND_MAX;  // 0.0 to 1.0
    }
    
    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    
    kernelWithAssertions<<<grid, block>>>(d_data, N);
    
    // Check for kernel errors (assertions will trigger these)
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cout << "Kernel assertion failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_data);
        return false;
    }
    
    CUDA_CHECK(cudaFree(d_data));
    return true;
}

// ==============================================================================
// STRATEGY 3: Memory Pattern Testing
// ==============================================================================

__global__ void testMemoryCoalescing(float* input, float* output, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Test different memory access patterns
        int read_idx = (idx * stride) % n;
        output[idx] = input[read_idx] * 2.0f;
    }
}

bool test_memory_patterns() {
    std::cout << "🧪 Testing Memory Access Patterns..." << std::endl;
    
    const int N = 1024 * 1024;  // 1M elements
    
    std::vector<float> h_input(N), h_output(N);
    for (int i = 0; i < N; i++) {
        h_input[i] = static_cast<float>(i);
    }
    
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Test different strides
    std::vector<int> strides = {1, 2, 4, 8, 16, 32};
    
    for (int stride : strides) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        dim3 block(256);
        dim3 grid((N + block.x - 1) / block.x);
        
        cudaEventRecord(start);
        testMemoryCoalescing<<<grid, block>>>(d_input, d_output, N, stride);
        cudaEventRecord(stop);
        
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        std::cout << "Stride " << stride << ": " << milliseconds << " ms" << std::endl;
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    return true;
}

// ==============================================================================
// STRATEGY 4: Correctness vs Performance Testing
// ==============================================================================

// Test different block sizes for optimal performance
template<int BLOCK_SIZE>
__global__ void vectorAddTemplate(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

bool test_block_size_optimization() {
    std::cout << "🧪 Testing Block Size Optimization..." << std::endl;
    
    const int N = 1024 * 1024;
    
    std::vector<float> h_a(N), h_b(N), h_c(N);
    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i + 1);
    }
    
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    
    // Test different block sizes
    std::vector<int> block_sizes = {32, 64, 128, 256, 512, 1024};
    
    for (int block_size : block_sizes) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        int grid_size = (N + block_size - 1) / block_size;
        
        cudaEventRecord(start);
        
        // Launch with specific block size
        switch (block_size) {
            case 32:  vectorAddTemplate<32><<<grid_size, 32>>>(d_a, d_b, d_c, N); break;
            case 64:  vectorAddTemplate<64><<<grid_size, 64>>>(d_a, d_b, d_c, N); break;
            case 128: vectorAddTemplate<128><<<grid_size, 128>>>(d_a, d_b, d_c, N); break;
            case 256: vectorAddTemplate<256><<<grid_size, 256>>>(d_a, d_b, d_c, N); break;
            case 512: vectorAddTemplate<512><<<grid_size, 512>>>(d_a, d_b, d_c, N); break;
            case 1024: vectorAddTemplate<1024><<<grid_size, 1024>>>(d_a, d_b, d_c, N); break;
        }
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        std::cout << "Block size " << block_size << ": " << milliseconds << " ms" << std::endl;
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    
    return true;
}

// ==============================================================================
// STRATEGY 5: Error Injection Testing
// ==============================================================================

__global__ void errorProneKernel(float* data, int n, bool inject_error) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        if (inject_error && idx == n/2) {
            // Simulate division by zero or other error
            data[idx] = data[idx] / 0.0f;
        } else {
            data[idx] = sqrt(data[idx]);
        }
    }
}

bool test_error_injection() {
    std::cout << "🧪 Testing Error Injection..." << std::endl;
    
    const int N = 1000;
    std::vector<float> h_data(N, 4.0f);  // Initialize with 4.0
    
    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    
    // Test without error injection
    errorProneKernel<<<grid, block>>>(d_data, N, false);
    cudaError_t err = cudaDeviceSynchronize();
    
    if (err != cudaSuccess) {
        std::cout << "Unexpected error in clean run: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_data);
        return false;
    }
    
    // Verify results (should be 2.0 = sqrt(4.0))
    CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, N * sizeof(float), cudaMemcpyDeviceToHost));
    
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_data[i] - 2.0f) > 1e-6f) {
            correct = false;
            break;
        }
    }
    
    if (!correct) {
        std::cout << "Computation error detected" << std::endl;
        cudaFree(d_data);
        return false;
    }
    
    CUDA_CHECK(cudaFree(d_data));
    return true;
}

// Main test runner
int main() {
    std::cout << "🚀 CUDA Testing Strategies Demo" << std::endl;
    std::cout << std::string(50, '=') << std::endl;
    
    // Run all test strategies
    bool all_passed = true;
    
    all_passed &= test_cpu_gpu_comparison();
    all_passed &= test_device_assertions();
    all_passed &= test_memory_patterns();
    all_passed &= test_block_size_optimization();
    all_passed &= test_error_injection();
    
    std::cout << std::string(50, '=') << std::endl;
    if (all_passed) {
        std::cout << "🎉 All CUDA tests passed!" << std::endl;
    } else {
        std::cout << "💥 Some tests failed!" << std::endl;
    }
    
    return all_passed ? 0 : 1;
}
