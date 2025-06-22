//
// Created by shane on 6/8/25.
//

#include "MultiLayerPerceptron.h"

void MultiLayerPerceptron::forward(std::vector<double> inputs) {
    for (double i : inputs) {
        layers.at(0).outputs.at(i) = inputs.at(i);
        std::cout << layers.at(0).outputs.at(i) << std::endl;
    }
}
