#!/bin/bash

# ============================================================================
# Dynamic Test Runner using C++ Dynamic Loading
# ============================================================================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="$PROJECT_ROOT/test"
TEST_RUNNER="$TEST_DIR/test_file_reader_dynamic"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Function to show usage
show_usage() {
    echo -e "${BLUE}🧪 Dynamic CUDA/C++ Test Runner${NC}"
    echo ""
    echo "Usage: $0 [TEST_FILES...]"
    echo ""
    echo "Arguments:"
    echo "  [no args]              Run all test files in test/ directory"
    echo "  TEST_FILES...          Run specific test files (relative to project root)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Run all tests"
    echo "  $0 test/tensor/test_basic_tensor.cu  # Run specific test"
    echo "  $0 test/tensor/*.cu                  # Run all tensor tests"
    echo ""
    echo "Note: This runner uses dynamic loading to discover and execute test functions."
    echo "Test functions must be declared with 'extern \"C\"' and follow 'bool test_*()' pattern."
    echo ""
}

# Function to build the test runner if needed
build_test_runner() {
    echo -e "${YELLOW}🔨 Building dynamic test runner...${NC}"
    
    cd "$TEST_DIR"
    if g++ test_file_reader_dynamic.cpp -ldl -I../include -o test_file_reader_dynamic; then
        echo -e "${GREEN}✅ Test runner built successfully${NC}"
        return 0
    else
        echo -e "${RED}❌ Failed to build test runner${NC}"
        return 1
    fi
}

# Function to check if test runner exists and is up to date
check_test_runner() {
    if [[ ! -f "$TEST_RUNNER" ]]; then
        echo -e "${YELLOW}⚠️  Test runner not found, building...${NC}"
        return 1
    fi
    
    # Check if source is newer than executable
    if [[ "$TEST_DIR/test_file_reader_dynamic.cpp" -nt "$TEST_RUNNER" ]]; then
        echo -e "${YELLOW}⚠️  Test runner source is newer, rebuilding...${NC}"
        return 1
    fi
    
    return 0
}

# Function to find all test files
find_all_test_files() {
    find "$TEST_DIR" -name "*.cu" -o -name "*.cpp" | grep -E "(test_|_test)" | sort
}

# Function to resolve test file paths
resolve_test_paths() {
    local resolved_paths=()
    
    for arg in "$@"; do
        # If path starts with /, treat as absolute
        if [[ "$arg" == /* ]]; then
            resolved_paths+=("$arg")
        # If path starts with test/, treat as relative to project root
        elif [[ "$arg" == test/* ]]; then
            resolved_paths+=("$PROJECT_ROOT/$arg")
        # Otherwise, treat as relative to current directory
        else
            resolved_paths+=("$(realpath "$arg" 2>/dev/null || echo "$arg")")
        fi
    done
    
    printf '%s\n' "${resolved_paths[@]}"
}

# Main execution
main() {
    echo -e "${BLUE}🚀 Dynamic CUDA/C++ Test Runner${NC}"
    echo -e "${BLUE}==================================${NC}"
    echo ""
    
    # Check and build test runner if needed
    if ! check_test_runner; then
        if ! build_test_runner; then
            echo -e "${RED}❌ Cannot proceed without test runner${NC}"
            exit 1
        fi
    fi
    
    cd "$PROJECT_ROOT"
    
    # Determine which tests to run
    if [[ $# -eq 0 ]]; then
        echo -e "${BLUE}🔍 Discovering all test files...${NC}"
        test_files=($(find_all_test_files))
        
        if [[ ${#test_files[@]} -eq 0 ]]; then
            echo -e "${RED}❌ No test files found in $TEST_DIR${NC}"
            exit 1
        fi
        
        echo -e "${GREEN}Found ${#test_files[@]} test files:${NC}"
        for file in "${test_files[@]}"; do
            echo "  • $(realpath --relative-to="$PROJECT_ROOT" "$file")"
        done
        echo ""
        
    else
        echo -e "${BLUE}🎯 Running specified test files...${NC}"
        
        # Resolve test file paths
        readarray -t test_files < <(resolve_test_paths "$@")
        
        # Validate test files exist
        valid_files=()
        for file in "${test_files[@]}"; do
            if [[ -f "$file" ]]; then
                valid_files+=("$file")
                echo -e "${GREEN}✓${NC} $(realpath --relative-to="$PROJECT_ROOT" "$file")"
            else
                echo -e "${RED}✗${NC} File not found: $file"
            fi
        done
        
        if [[ ${#valid_files[@]} -eq 0 ]]; then
            echo -e "${RED}❌ No valid test files found${NC}"
            exit 1
        fi
        
        test_files=("${valid_files[@]}")
        echo ""
    fi
    
    # Convert absolute paths to relative paths for the test runner
    relative_test_files=()
    for file in "${test_files[@]}"; do
        relative_path=$(realpath --relative-to="$TEST_DIR" "$file")
        relative_test_files+=("$relative_path")
    done
    
    # Run the dynamic test runner
    echo -e "${YELLOW}🧪 Executing dynamic test runner...${NC}"
    echo ""
    
    cd "$TEST_DIR"
    if ./test_file_reader_dynamic "${relative_test_files[@]}"; then
        echo ""
        echo -e "${GREEN}🎉 Test execution completed successfully!${NC}"
        exit 0
    else
        echo ""
        echo -e "${RED}💥 Test execution failed!${NC}"
        exit 1
    fi
}

# Handle help flag
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    show_usage
    exit 0
fi

# Run main function
main "$@"