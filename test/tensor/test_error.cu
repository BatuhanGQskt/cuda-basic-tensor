#include <iostream>
#include <stdexcept>
#include "../../include/tensor/basic_tensor.h"

extern "C" bool test_error_handling() {
    std::cout << "Testing Error Handling" << std::endl;
    std::cout << "--------------------------------" << std::endl;
    
    return false;
}

extern "C" bool test_error_handling_2() {
    std::cout << "Testing Error Handling 2" << std::endl;
    std::cout << "--------------------------------" << std::endl;
    
    return true;
}