#include "MultiLayerPerceptron.h"

void MultiLayerPerceptron::forwardPass(const std::vector<double> &inputs) {
    for (size_t i = 0; i < inputs.size(); i++) {
        layers[0].outputs[i] = inputs[i];
    }
    double weightSum = 0;
    for (int l = 1; l < layers.size(); l++) { // Each layer
        for (int i = 0; i < layers[l].numOfNeurons; i++) { // This layer's neurons
            weightSum = 0;
            for (int j = 0; j < layers[l - 1].numOfNeurons; j++) { // Previous layer's neurons
                weightSum += layers[l - 1].outputs[j] * layers[l].weights[i * layers[l - 1].numOfNeurons + j];
            }
            weightSum += layers[l].biases[i];
            layers[l].preActivations[i] = weightSum;
            layers[l].outputs[i] = activationFunction->calculate(weightSum);
        }
    }
}

double MultiLayerPerceptron::backwardsPass(const std::vector<double> &outputs, const double learningRate) {
    double totalError = 0;
    double error = 0;
    double delta;

    for (int i = layers.size() - 1; i > 0; i--) {
        if (i == layers.size() - 1) { // Output layer
            for (int j = 0; j < layers[layers.size() - 1].numOfNeurons; j++) {
                error = outputs[j] - layers[layers.size() - 1].outputs[j];
                totalError += error * error;
                delta = error * activationFunction->calculateDerivative(layers.at(layers.size() - 1).preActivations.at(j));
                layers[i].updateWeights(delta, &layers[i - 1], learningRate, j);
            }
        } else {
            for (int j = 0; j < layers[i].numOfNeurons; j++) {
                delta = 0;
                for (int k = 0; k < layers[i + 1].numOfNeurons; k++) {
                    delta += layers[i + 1].deltas[k] * layers[i + 1].weights[k * layers[i].numOfNeurons + j];
                }
                delta *= activationFunction->calculateDerivative(layers[i].preActivations[j]);
                layers[i].updateWeights(delta, &layers[i - 1], learningRate, j);
            }
        }
    }
    error = totalError / layers[layers.size() - 1 ].numOfNeurons;
    //std::cout << "CURRENT ERROR: " << error << std::endl;
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
            //std::cout << totalError << std::endl;
        }
        if (i % 100 == 0)
        {
            std::cout << "Epoch #" << i << ", Current error: " << totalError << std::endl;
        }
    }
}

void MultiLayerPerceptron::testOutputs(MultiLayerPerceptron &mlp, const std::vector<std::vector<double>> &inputs, const std::vector<std::vector<double>> &outputs) {
    int correctOutputs = 0;
    double difference = 0;
    for (int i = 0; i < inputs.size(); i++)
    {
        mlp.forwardPass(inputs[i]);
        difference = fabs(mlp.layers[mlp.numOfLayers - 1].outputs[0] - outputs[i][0]);
        if (constexpr double acceptableError = 0.01; difference <= acceptableError)
        {
            correctOutputs++;
        }
        /*std::cout << std::round(mlp.layers[mlp.numOfLayers - 1].outputs[0] * 100.0) / 100.0 << ", " << std::round(outputs[i][0] * 100.0) / 100.0 << std::endl;
        if (std::round(mlp.layers[mlp.numOfLayers - 1].outputs[0] * 100.0) / 100.0 == std::round(outputs[i][0] * 100.0) / 100.0) {
            correctOutputs++;
        }*/
    }
    std::cout << "Correct outputs: " << correctOutputs << " / " << inputs.size() << std::endl;

    /*bool correct = true;
    std::cout << "testOutputs running... " << std::endl;
    for (int i = 0; i < inputs.size(); i++) {
        mlp.forwardPass(inputs[i]);
        for (int j = 0; j < outputs[i].size(); j++) {
            std::cout << mlp.layers[mlp.numOfLayers - 1].outputs[j] << std::endl;
            if (mlp.layers[mlp.numOfLayers - 1].outputs[j] != outputs[i][j]) {
                correct = false;
            }
        }
    }*/
}
