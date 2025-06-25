//
// Created by shane on 6/8/25.
//

#ifndef LAYER_H
#define LAYER_H
#include <cstdlib>
#include <iostream>
#include <vector>

class Layer {
public:
    int numInputs;
    int numOfNeurons;

    std::vector<double> weights;
    std::vector<double> biases;
    std::vector<double> preActivations;
    std::vector<double> outputs;

    Layer(const int numInputs, const int numOfNeurons):
        numInputs(numInputs), numOfNeurons(numOfNeurons),
        weights(numInputs * numOfNeurons), biases(numOfNeurons), preActivations(numOfNeurons), outputs(numOfNeurons)
    {
        //srand(time(nullptr));
        for (int i = 0; i < numInputs * numOfNeurons; i++) // Doesn't need to be truly random, just need neurons to have different values
        {
            weights[i] = static_cast<double>(rand())/RAND_MAX*2.0-1.0;
        }
        for (int i = 0; i < numOfNeurons; i++)
        {
            biases[i] = static_cast<double>(rand())/RAND_MAX*2.0-1.0;
            outputs[i] = 0;
            preActivations[i] = 0;
        }
    }

    double& weight(const int inputId, const int neuronId)
    {
        return weights[neuronId * numInputs + inputId];
    }

    void updateWeights(double delta, const Layer* previousLayer, double learningRate) {
        for (int i = 0; i < previousLayer->numOfNeurons; i++) {
            weights[i] += learningRate * delta * previousLayer->outputs[i];
        }
    }
};

#endif //LAYER_H