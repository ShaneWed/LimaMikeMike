#ifndef TRAININGDATA_H
#define TRAININGDATA_H
#include <vector>

class TrainingData {
public:
    int numInputs;
    int numOutputs;
    int numTrainingVectors;
    int numTestingVectors;
    std::vector<std::vector<double>> trainingInputs;
    std::vector<std::vector<double>> trainingOutputs;
    std::vector<std::vector<double>> testingInputs;
    std::vector<std::vector<double>> testingOutputs;

    TrainingData(const int numInputs, const int numOutputs, const int numTrainingVectors, const int numTestingVectors):
    numInputs(numInputs), numOutputs(numOutputs), numTrainingVectors(numTrainingVectors), numTestingVectors(numTestingVectors),
    trainingInputs(numTrainingVectors), trainingOutputs(numTrainingVectors), testingInputs(numTestingVectors), testingOutputs(numTestingVectors)
    {

    }
};

#endif //TRAININGDATA_H
