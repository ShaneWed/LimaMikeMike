#ifndef LAYER_H
#define LAYER_H
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
        constexpr int min = -1;
        constexpr int max = 1;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> range(min, max);
        for (int i = 0; i < numInputs * numOfNeurons; i++) // Doesn't need to be truly random, just need neurons to have different values
        {
            weights[i] = range(gen);
            //std::cout << weights[i] << std::endl;
        }
        for (int i = 0; i < numOfNeurons; i++)
        {
            //biases[i] = static_cast<double>(rand())/RAND_MAX*0.1-0.05;
            biases[i] = range(gen);
            outputs[i] = 0;
            preActivations[i] = 0;
            deltas[i] = 0;
        }
    }

    // Also updated biases
    void updateWeights(double delta, const Layer* previousLayer, double learningRate, int neuron);
};

#endif //LAYER_H