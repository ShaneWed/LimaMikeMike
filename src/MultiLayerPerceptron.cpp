//
// Created by shane on 6/8/25.
//

#include "MultiLayerPerceptron.h"

void MultiLayerPerceptron::forward(std::vector<double> inputs) {
    for (double i : inputs) {
        layers.at(0).outputs.at(i) = inputs.at(i);
        std::cout << layers.at(0).outputs.at(i) << std::endl;
    }
    double weightSum = 0;
    for (int l = 1; l < layers.size(); l++) { // Each layer
        for (int i = 0; i < layers.at(l).numOfNeurons; i++) { // This layer's neurons
            weightSum = 0;
            for (int j = 0; j < layers.at(l - 1).numOfNeurons; j++) { // Previous layer's neurons
                weightSum += layers.at(l - 1).outputs.at(j) * layers.at(l).weights.at(i);
            }
            weightSum += layers.at(l).biases.at(i);
            layers.at(l).weights.at(i) = activationFunction->calculate(weightSum);
            std::cout << activationFunction->calculate(weightSum) << std::endl;
        }
    }
    std::cout << "Forward pass complete!" << std::endl;
}

double MultiLayerPerceptron::backwards(std::vector<double> outputs, double learningRate) {
    double error = 0;
    double delta;

    for (int i = layers.size() - 1; i > 0; i--) {
        if (i == layers.size() - 1) { // Output layer
            for (int j = 0; j < layers.at(layers.size() - 1).numOfNeurons; j++) {
                error = outputs.at(j) - layers.at(layers.size() - 1).outputs.at(j);
                delta = error * activationFunction->calculateDerivative(layers.at(layers.size() - 1).outputs.at(j)); // May have to change to PreActivation like in MDIC
                
            }
        }
    }

    return 0.0;
}

