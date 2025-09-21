#include <iostream>
#include "MultiLayerPerceptron.h"
#include "TrainingData.h"

int main()
{
    /*const std::vector<std::vector<double>> input = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    const std::vector<std::vector<double>> output = {{0}, {1}, {1}, {0}};
    const std::vector<std::vector<double>> incorrectOutput = {{-1}, {-1}, {-1}, {-1}};
    Sigmoid sigmoid;
    Tanh tanh;
    ReLU relu;
    MultiLayerPerceptron limaMikeMike(2, 5, 1, 3, 0.6, &sigmoid);
    MultiLayerPerceptron::testOutputs(limaMikeMike, input, output);
    MultiLayerPerceptron::train(limaMikeMike, input, output, 1000);
    std::cout << "Training complete" << std::endl;
    MultiLayerPerceptron::testOutputs(limaMikeMike, input, output);
    std::cout << "Success!" << std::endl;
    return 0;*/

    Tanh tanh;
    MultiLayerPerceptron limaMikeMike(4, 5, 1, 10, 0.1, &tanh);
    const TrainingData sinData(4, 1, 1000, 100);
    //MultiLayerPerceptron::testOutputs(limaMikeMike, sinData.testingInputs, sinData.testingOutputs);
    MultiLayerPerceptron::train(limaMikeMike, sinData.trainingInputs, sinData.trainingOutputs, 1000);
    MultiLayerPerceptron::testOutputs(limaMikeMike, sinData.testingInputs, sinData.testingOutputs);
    std::cout << "Success!" << std::endl;
    MultiLayerPerceptron::testOutputs(limaMikeMike, sinData.trainingInputs, sinData.trainingOutputs);
    return 0;
}
