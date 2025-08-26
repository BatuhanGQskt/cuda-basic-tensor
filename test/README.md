# Test Directory

This directory contains all testing infrastructure and test files for the CUDA/C++ project.

## Structure

```
test/
├── README.md                    # This file
├── CMakeLists.txt              # CMake configuration for tests
├── run_tests_simple.sh         # Simple shell-based test runner
├── run_tests.cmake             # CMake script integration
├── build/                      # Build directory (created when using CMake)
├── tensor/                     # Tensor-related tests
│   ├── test_basic_tensor.cu
│   └── simple_working_test.cu
└── *.cu                        # Other test files
```

## Running Tests

### Method 1: Simple Script (Recommended for Development)

```bash
# From project root
./run_tests.sh                           # Run all tests
./run_tests.sh --pattern tensor          # Run tests matching pattern

# Or directly from test directory
cd test
./run_tests_simple.sh                    # Run all tests
./run_tests_simple.sh tensor/test_basic_tensor.cu  # Run specific test
./run_tests_simple.sh --pattern tensor   # Run pattern matching
```

### Method 2: CMake + CTest (Recommended for CI/CD)

```bash
# From test directory
cd test
mkdir build && cd build
cmake ..
make
ctest --output-on-failure

# Run specific tests
ctest -R tensor                # Run tests matching "tensor"
ctest -R test_basic           # Run tests matching "test_basic"

# Use custom targets
make run_all_tests            # Run all tests
make run_tensor_tests         # Run tensor tests only
make run_tests_verbose        # Run with verbose output
```

## Adding New Tests

1. Create a new `.cu` or `.cpp` file in the appropriate subdirectory
2. Include necessary headers: `#include "../../include/your_header.h"`
3. Write test functions that return `bool` (true for pass, false for fail)
4. Add a `main()` function that calls your test functions

Example test structure:
```cpp
#include <iostream>
#include "../../include/tensor/basic_tensor.h"

bool test_my_feature() {
    // Your test code here
    if (/* test condition */) {
        std::cout << "✅ Test passed" << std::endl;
        return true;
    } else {
        std::cout << "❌ Test failed" << std::endl;
        return false;
    }
}

int main() {
    std::cout << "🚀 Testing My Feature" << std::endl;
    
    bool success = test_my_feature();
    
    return success ? 0 : 1;
}
```

## Test Categories

- **tensor/**: Tests for tensor operations and data structures
- **cuda/**: CUDA-specific tests (kernels, memory management, etc.)
- **integration/**: End-to-end integration tests
- **performance/**: Benchmarking and performance tests

## Guidelines

1. **Name tests descriptively**: `test_tensor_creation.cu`, `test_matrix_multiplication.cu`
2. **Keep tests focused**: One test file per feature/class
3. **Use clear output**: Include ✅/❌ indicators and descriptive messages
4. **Test edge cases**: Empty inputs, large inputs, error conditions
5. **Clean up resources**: Free memory, reset state between tests

## Debugging Tests

- Add `-g` flag for debug symbols: modify the compile commands in scripts
- Use `cuda-gdb` for CUDA debugging
- Add verbose output with detailed error messages
- Use `valgrind` for memory leak detection (CPU code only)

## Environment

- CUDA Version: 11.5+
- C++ Standard: C++11
- Supported GPU Architectures: 75, 80, 86
- Required: nvcc compiler, CMake 3.18+
