//
// Created by shane on 6/8/25.
//

#ifndef MULTILAYERPERCEPTRON_H
#define MULTILAYERPERCEPTRON_H

#include <iostream>
#include "Layer.h"

class MultiLayerPerceptron {
public:
    int numOfInputs;
    int numOfHiddenNeurons;
    int numOfOutputs;
    int numOfLayers;
    double learningRate;
    static std::vector<Layer> layers;

    MultiLayerPerceptron(const int numOfInputs, const int numOfHiddenNeurons, const int numOfOutputs, const int numOfLayers, const double learningRate):
        numOfInputs(numOfInputs), numOfHiddenNeurons(numOfHiddenNeurons), numOfOutputs(numOfOutputs),
        numOfLayers(numOfLayers), learningRate(learningRate)
    {
        layers = {Layer(numOfInputs, numOfHiddenNeurons)};
        for (int i = 1; i < numOfLayers - 1; i++)
        {
            layers.emplace_back(numOfHiddenNeurons, numOfHiddenNeurons);
        }
        layers.emplace_back(numOfHiddenNeurons, numOfOutputs);
        std::cout << "MultiLayerPerceptron successfully created!" << std::endl;
        std::cout << layers.size() << std::endl;
    }

    static void forward(double inputs[]);
};

#endif //MULTILAYERPERCEPTRON_H
