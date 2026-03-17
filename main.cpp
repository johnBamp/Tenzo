#include <iostream>
#include "include/tensor.hpp"

int main() {
    tenzo::Tensor A({2, 3});
    tenzo::Tensor B({3, 2});

    A = {
        1, 2, 3,
        4, 5, 6
    };

    B = {
        7,  8,
        9, 10,
        11, 12
    };

    tenzo::Tensor C = A * B;

    std::cout << C << std::endl;
        
    return 0;
}