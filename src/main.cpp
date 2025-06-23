#include <iostream>
#include "MultiLayerPerceptron.h"

int main()
{
    Tanh tanh;
    MultiLayerPerceptron limaMikeMike(2, 5, 1, 3, 0.05, &tanh);
    limaMikeMike.forward(std::vector<double>{1, 0});
    std::cout << "Success!" << std::endl;
    return 0;
}
