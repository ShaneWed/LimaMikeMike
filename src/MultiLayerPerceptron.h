//
// Created by shane on 6/8/25.
//

#ifndef MULTILAYERPERCEPTRON_H
#define MULTILAYERPERCEPTRON_H

#include <iostream>

#include "ActivationFunction.h"
#include "Layer.h"

class MultiLayerPerceptron {
public:
    int numOfInputs;
    int numOfHiddenNeurons;
    int numOfOutputs;
    int numOfLayers;
    double learningRate;
    std::vector<Layer> layers;
    ActivationFunction* activationFunction;

    MultiLayerPerceptron(const int numOfInputs, const int numOfHiddenNeurons, const int numOfOutputs, const int numOfLayers, const double learningRate, ActivationFunction* activationFunction):
        numOfInputs(numOfInputs), numOfHiddenNeurons(numOfHiddenNeurons), numOfOutputs(numOfOutputs),
        numOfLayers(numOfLayers), learningRate(learningRate), activationFunction(activationFunction)
    {
        layers = {Layer(numOfInputs, numOfInputs)};
        for (int i = 1; i < numOfLayers - 1; i++)
        {
            layers.emplace_back(numOfHiddenNeurons, numOfHiddenNeurons);
        }
        layers.emplace_back(numOfHiddenNeurons, numOfOutputs);
        std::cout << "MultiLayerPerceptron successfully created!" << std::endl;
        //std::cout << layers.size() << std::endl;
    }

    void forwardPass(std::vector<double> inputs);
    double backwardsPass(const std::vector<double> &outputs, double learningRate);
    static void train(MultiLayerPerceptron &mlp, const std::vector<std::vector<double>> &inputs, const std::vector<std::vector<double>> &outputs, int epochs);
    static void testOutputs(MultiLayerPerceptron &mlp, const std::vector<std::vector<double>> &inputs, const std::vector<std::vector<double>> &outputs);
};

#endif //MULTILAYERPERCEPTRON_H
