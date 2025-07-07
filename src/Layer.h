//
// Created by shane on 6/8/25.
//

#ifndef LAYER_H
#define LAYER_H
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

class Layer {
public:
    int numInputs;
    int numOfNeurons;

    std::vector<double> weights;
    std::vector<double> biases;
    std::vector<double> deltas;
    std::vector<double> preActivations;
    std::vector<double> outputs;

    Layer(const int numInputs, const int numOfNeurons):
        numInputs(numInputs), numOfNeurons(numOfNeurons),
        weights(numInputs * numOfNeurons), biases(numOfNeurons), deltas(numOfNeurons), preActivations(numOfNeurons), outputs(numOfNeurons)
    {
        std::default_random_engine random{std::random_device{}()};
        std::uniform_real_distribution<> range(-1, 1);
        //srand(time(nullptr));
        for (int i = 0; i < numInputs * numOfNeurons; i++) // Doesn't need to be truly random, just need neurons to have different values
        {
            weights[i] = range(random);
        }
        for (int i = 0; i < numOfNeurons; i++)
        {
            //biases[i] = static_cast<double>(rand())/RAND_MAX*0.1-0.05;
            biases[i] = range(random);
            outputs[i] = 0;
            preActivations[i] = 0;
            deltas[i] = 0;
        }
    }

    // Also updated biases
    void updateWeights(double delta, const Layer* previousLayer, double learningRate, int neuron) {
        deltas.at(neuron) = delta;
        for (int i = 0; i < previousLayer->numInputs; i++) {
            //weights[i] += learningRate * delta * previousLayer->outputs[i];
            weights[neuron * previousLayer->numInputs + i] += learningRate * delta * previousLayer->outputs[i];
        }
        biases.at(neuron) = biases.at(neuron) + learningRate * delta;
    }
};

#endif //LAYER_H