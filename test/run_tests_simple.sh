#!/bin/bash

# ============================================================================
# Simple CUDA/C++ Test Runner
# ============================================================================

TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$TEST_DIR/.." && pwd)"
BUILD_DIR="$TEST_DIR/build"
INCLUDE_DIR="$PROJECT_ROOT/include"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Create build directory
mkdir -p "$BUILD_DIR"

# Function to compile and run a test
run_test() {
    local test_file="$1"
    local test_name=$(basename "$test_file" .cu)
    local executable="$BUILD_DIR/$test_name"
    
    echo -e "${YELLOW}🔨 Compiling: $test_file${NC}"
    
    # Compile
    if nvcc -I"$INCLUDE_DIR" "$test_file" -o "$executable"; then
        echo -e "${GREEN}✅ Compilation successful${NC}"
        
        echo -e "${BLUE}🧪 Running: $test_name${NC}"
        if "$executable"; then
            echo -e "${GREEN}✅ Test PASSED: $test_name${NC}"
            return 0
        else
            echo -e "${RED}❌ Test FAILED: $test_name${NC}"
            return 1
        fi
    else
        echo -e "${RED}❌ Compilation FAILED: $test_file${NC}"
        return 1
    fi
}

# Function to show usage
show_usage() {
    echo -e "${BLUE}🧪 Simple CUDA Test Runner${NC}"
    echo ""
    echo "Usage: $0 [OPTIONS] [TEST_FILES...]"
    echo ""
    echo "Options:"
    echo "  --all, -a          Run all tests"
    echo "  --pattern, -p PAT  Run tests matching pattern"
    echo "  --help, -h         Show this help"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Run all tests"
    echo "  $0 --all                             # Run all tests"
    echo "  $0 test/tensor/test_basic_tensor.cu  # Run specific test"
    echo "  $0 --pattern tensor                  # Run tests with 'tensor' in name"
    echo ""
}

# Main logic
if [[ $# -eq 0 ]] || [[ "$1" == "--all" ]] || [[ "$1" == "-a" ]]; then
    # Run all tests
    echo -e "${BLUE}🚀 Running all tests in $TEST_DIR${NC}"
    
    test_files=$(find "$TEST_DIR" -name "*.cu" -type f)
    total_tests=0
    passed_tests=0
    
    for test_file in $test_files; do
        ((total_tests++))
        if run_test "$test_file"; then
            ((passed_tests++))
        fi
        echo ""
    done
    
    echo -e "${YELLOW}📊 Results: $passed_tests/$total_tests tests passed${NC}"
    
    if [[ $passed_tests -eq $total_tests ]]; then
        echo -e "${GREEN}🎉 All tests passed!${NC}"
        exit 0
    else
        echo -e "${RED}💥 Some tests failed!${NC}"
        exit 1
    fi
    
elif [[ "$1" == "--pattern" ]] || [[ "$1" == "-p" ]]; then
    # Run tests matching pattern
    if [[ -z "$2" ]]; then
        echo -e "${RED}❌ Pattern required${NC}"
        show_usage
        exit 1
    fi
    
    pattern="$2"
    echo -e "${BLUE}🔍 Running tests matching pattern: $pattern${NC}"
    
    test_files=$(find "$TEST_DIR" -name "*$pattern*.cu" -type f)
    
    if [[ -z "$test_files" ]]; then
        echo -e "${RED}❌ No tests found matching pattern: $pattern${NC}"
        exit 1
    fi
    
    total_tests=0
    passed_tests=0
    
    for test_file in $test_files; do
        ((total_tests++))
        if run_test "$test_file"; then
            ((passed_tests++))
        fi
        echo ""
    done
    
    echo -e "${YELLOW}📊 Results: $passed_tests/$total_tests tests passed${NC}"
    
elif [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    show_usage
    exit 0
    
else
    # Run specific test files
    echo -e "${BLUE}🎯 Running specific tests${NC}"
    
    total_tests=0
    passed_tests=0
    
    for test_file in "$PROJECT_ROOT/$@"; do
        if [[ ! -f "$test_file" ]]; then
            echo -e "${RED}❌ Test file not found: $test_file${NC}"
            continue
        fi
        
        ((total_tests++))
        if run_test "$test_file"; then
            ((passed_tests++))
        fi
        echo ""
    done
    
    echo -e "${YELLOW}📊 Results: $passed_tests/$total_tests tests passed${NC}"
    
    if [[ $passed_tests -eq $total_tests ]]; then
        exit 0
    else
        exit 1
    fi
fi
