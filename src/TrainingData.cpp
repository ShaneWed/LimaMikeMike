#include "TrainingData.h"
#include <fstream>
#include <iostream>

void TrainingData::initializeLetters() {
    std::ifstream inputFile("Letters.txt");
    std::string inputString;

    for (int i = 0; i < 10; i++) {
        getline(inputFile, inputString);
        std::cout << inputString << std::endl;
    }
}
