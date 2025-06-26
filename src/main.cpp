#include <iostream>
#include "MultiLayerPerceptron.h"

int main()
{
    const std::vector<std::vector<double>> input = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    const std::vector<std::vector<double>> output = {{0}, {1}, {1}, {0}};
    Tanh tanh;
    MultiLayerPerceptron limaMikeMike(2, 5, 1, 3, 0.05, &tanh); // Seg fault here
    //MultiLayerPerceptron::train(limaMikeMike, input, output, 100);
    std::cout << "Success!" << std::endl;
    return 0;
}
