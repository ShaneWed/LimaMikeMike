#include "Layer.h"

void Layer::updateWeights(double delta, const Layer* previousLayer, double learningRate, int neuron) {
    deltas[neuron] = delta;
    for (int i = 0; i < previousLayer->numOfNeurons; i++) {
        //weights[i] += learningRate * delta * previousLayer->outputs[i];
        weights[neuron * previousLayer->numOfNeurons + i] += learningRate * delta * previousLayer->outputs[i];
    }
    biases[neuron] = biases[neuron] + learningRate * delta;
}