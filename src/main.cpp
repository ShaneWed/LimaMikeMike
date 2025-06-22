#include <iostream>
#include "MultiLayerPerceptron.h"

int main()
{
    MultiLayerPerceptron limaMikeMike(2, 5, 1, 3, 0.05);
    limaMikeMike.forward(std::vector<double>{1, 0});
    std::cout << "Success!" << std::endl;
    return 0;
}
