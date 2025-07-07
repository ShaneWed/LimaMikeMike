//
// Created by shane on 6/8/25.
//

#include "Layer.h"

void Layer::updateWeights(double delta, const Layer* previousLayer, double learningRate, int neuron) {
    deltas.at(neuron) = delta;
    for (int i = 0; i < previousLayer->numInputs; i++) {
        //weights[i] += learningRate * delta * previousLayer->outputs[i];
        weights[neuron * previousLayer->numInputs + i] += learningRate * delta * previousLayer->outputs[i];
    }
    biases.at(neuron) = biases.at(neuron) + learningRate * delta;
}