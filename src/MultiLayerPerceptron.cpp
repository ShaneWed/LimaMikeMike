//
// Created by shane on 6/8/25.
//

#include "MultiLayerPerceptron.h"

void MultiLayerPerceptron::forwardPass(std::vector<double> inputs) {
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
            layers.at(l).preActivations.at(i) = weightSum;
            layers.at(l).outputs.at(i) = activationFunction->calculate(weightSum);
            std::cout << activationFunction->calculate(weightSum) << std::endl;
        }
    }
    std::cout << "Forward pass complete!" << std::endl;
}

double MultiLayerPerceptron::backwardsPass(const std::vector<double> &outputs, double learningRate) {
    double error = 0;
    double delta;

    for (int i = layers.size() - 1; i > 0; i--) {
        if (i == layers.size() - 1) { // Output layer
            for (int j = 0; j < layers.at(layers.size() - 1).numOfNeurons; j++) {
                error = outputs.at(j) - layers.at(layers.size() - 1).outputs.at(j);
                delta = error * activationFunction->calculateDerivative(layers.at(layers.size() - 1).preActivations.at(j));
                layers.at(i).updateWeights(delta, &layers.at(i - 1), learningRate, j);
            }
        } else {
            for (int j = 0; j < layers.at(i).numOfNeurons; j++) {
                delta = 0;
                for (int k = 0; k < layers.at(i + 1).numOfNeurons; k++) {
                    delta += layers.at(i + 1).deltas.at(k) * layers.at(i + 1).weights.at(j);
                }
                delta *= activationFunction->calculateDerivative(layers.at(i).preActivations.at(j));
                layers.at(i).updateWeights(delta, &layers.at(i - 1), learningRate, j);
            }
        }
    }
    error = error / layers.at(layers.size() - 1).numOfNeurons;
    return error;
}

void MultiLayerPerceptron::train(MultiLayerPerceptron &mlp, const std::vector<std::vector<double>> &inputs, const std::vector<std::vector<double>> &outputs, int epochs) {
    double totalError = 0;
    double error = 0;
    for (int i = 0; i < epochs; i++) {
        totalError = 0;
        for (int j = 0; j < inputs.size(); j++) {
            mlp.forwardPass(inputs[j]);
            error = mlp.backwardsPass(outputs[j], mlp.learningRate);
            totalError += fabs(error);
        }
    }
}
