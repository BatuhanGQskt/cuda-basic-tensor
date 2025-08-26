#include <iostream>

extern "C" bool test_dummy_success() {
    std::cout << "Testing Dummy Success" << std::endl;
    std::cout << "--------------------------------" << std::endl;
    
    return true;
}

extern "C" bool test_dummy_failure() {
    std::cout << "Testing Dummy Failure" << std::endl;
    std::cout << "--------------------------------" << std::endl;
    
    return false;
}