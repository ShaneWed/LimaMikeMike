#include <iostream>
#include "MultiLayerPerceptron.h"

int main()
{
    const std::vector<std::vector<double>> input = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    const std::vector<std::vector<double>> output = {{0}, {1}, {1}, {0}};
    Tanh tanh;
    MultiLayerPerceptron limaMikeMike(2, 5, 1, 3, 0.05, &tanh);
    MultiLayerPerceptron::train(limaMikeMike, input, output, 1);
    MultiLayerPerceptron::testOutputs(limaMikeMike, input, output);
    std::cout << "Success!" << std::endl;
    return 0;
}
