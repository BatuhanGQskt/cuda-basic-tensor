#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cuda_runtime.h>
#include <iostream>

/**
 * @brief CUDA error checking macro
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

/**
 * @brief GPU device information
 */
struct GPUInfo {
    int deviceCount;
    int currentDevice;
    size_t totalMemory;
    size_t freeMemory;
    int multiProcessorCount;
    int maxThreadsPerBlock;
    
    __host__ static GPUInfo getInfo();
    __host__ void print() const;
};

/**
 * @brief GPU memory management utilities
 */
template<typename T>
class CudaArray {
private:
    T* d_ptr;
    size_t size;
    
public:
    __host__ CudaArray(size_t count);
    __host__ ~CudaArray();
    
    // Disable copy constructor and assignment
    CudaArray(const CudaArray&) = delete;
    CudaArray& operator=(const CudaArray&) = delete;
    
    // Move constructor and assignment
    __host__ CudaArray(CudaArray&& other) noexcept;
    __host__ CudaArray& operator=(CudaArray&& other) noexcept;
    
    __host__ T* get() const { return d_ptr; }
    __host__ size_t getSize() const { return size; }
    
    __host__ void copyFromHost(const T* h_ptr);
    __host__ void copyToHost(T* h_ptr);
    __host__ void memset(int value = 0);
};

/**
 * @brief Kernel launch configuration helper
 */
struct KernelConfig {
    dim3 blockSize;
    dim3 gridSize;
    size_t sharedMemSize;
    cudaStream_t stream;
    
    __host__ KernelConfig(int totalThreads, int preferredBlockSize = 256);
    __host__ KernelConfig(dim3 grid, dim3 block, size_t sharedMem = 0, cudaStream_t s = 0);
};

/**
 * @brief Performance timing utilities
 */
class CudaTimer {
private:
    cudaEvent_t start, stop;
    bool timing;
    
public:
    __host__ CudaTimer();
    __host__ ~CudaTimer();
    
    __host__ void startTiming();
    __host__ float stopTiming(); // Returns elapsed time in milliseconds
};

/**
 * @brief Common CUDA kernel templates
 */

// Template for element-wise operations
template<typename T, typename Op>
__global__ void elementWiseOp(T* a, T* b, T* result, int N, Op operation) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        result[idx] = operation(a[idx], b[idx]);
    }
}

// Template for reductions
template<typename T>
__global__ void reduce(T* input, T* output, int N) {
    extern __shared__ T sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (idx < N) ? input[idx] : 0;
    __syncthreads();
    
    // Perform reduction in shared memory
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2*s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// Template for parallel for loops
template<typename Func>
__global__ void parallelFor(int N, Func func) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        func(idx);
    }
}

#endif // CUDA_UTILS_CUH
