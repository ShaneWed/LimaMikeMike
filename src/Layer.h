//
// Created by shane on 6/8/25.
//

#ifndef LAYER_H
#define LAYER_H
#include <vector>

class Layer {
public:
    int numInputs;
    int numOfNeurons;

    std::vector<double> weights;
    std::vector<double> biases;
    std::vector<double> outputs;

    Layer(const int numInputs, const int numOfNeurons):
        numInputs(numInputs), numOfNeurons(numOfNeurons),
        weights(numInputs * numOfNeurons), biases(numOfNeurons), outputs(numOfNeurons) {}

    double& weight(const int inputId, const int neuronId)
    {
        return weights[neuronId * numInputs + inputId];
    }
};

#endif //LAYER_H