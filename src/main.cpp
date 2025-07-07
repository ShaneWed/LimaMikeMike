#include <iostream>
#include "MultiLayerPerceptron.h"

int main()
{
    const std::vector<std::vector<double>> input = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    const std::vector<std::vector<double>> output = {{0}, {1}, {1}, {0}};
    const std::vector<std::vector<double>> incorrectOutput = {{-1}, {-1}, {-1}, {-1}};
    Sigmoid sigmoid;
    Tanh tanh;
    ReLU relu;
    MultiLayerPerceptron limaMikeMike(2, 5, 1, 3, 0.01, &relu);
    MultiLayerPerceptron::testOutputs(limaMikeMike, input, output);
    MultiLayerPerceptron::train(limaMikeMike, input, output, 1000);
    std::cout << "Training complete" << std::endl;
    MultiLayerPerceptron::testOutputs(limaMikeMike, input, output);
    std::cout << "Success!" << std::endl;
    return 0;
}
