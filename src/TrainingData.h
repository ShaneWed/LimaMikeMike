#ifndef TRAININGDATA_H
#define TRAININGDATA_H
#include <random>
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
    std::vector<std::vector<double>> letterTrainingInputs;
    std::vector<std::vector<double>> letterTrainingOutputs;
    std::vector<std::vector<double>> letterTestingInputs;
    std::vector<std::vector<double>> letterTestingOutputs;

    TrainingData(const int numInputs, const int numOutputs, const int numTrainingVectors, const int numTestingVectors):
    numInputs(numInputs), numOutputs(numOutputs), numTrainingVectors(numTrainingVectors), numTestingVectors(numTestingVectors),
    trainingInputs(numTrainingVectors, std::vector<double>(numInputs)),
    trainingOutputs(numTrainingVectors, std::vector<double>(numOutputs)),
    testingInputs(numTestingVectors, std::vector<double>(numInputs)),
    testingOutputs(numTestingVectors, std::vector<double>(numOutputs))
    {
        int min = -1;
        int max = 1;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> range(min, max);
        for (int i = 0; i < numTrainingVectors; i++)
        {
            for (int j = 0; j < numInputs; j++)
            {
                trainingInputs[i][j] = range(gen);
            }
            double temp = trainingInputs[i][0] + trainingInputs[i][1] - trainingInputs[i][2] + trainingInputs[i][3];
            trainingOutputs[i][0] = sin(temp);
        }
        for (int i = 0; i < numTestingVectors; i++)
        {
            for (int j = 0; j < numInputs; j++)
            {
                testingInputs[i][j] = range(gen);
            }
            double temp = testingInputs[i][0] + testingInputs[i][1] - testingInputs[i][2] + testingInputs[i][3];
            testingOutputs[i][0] = sin(temp);
        }
    }

    static void initializeLetters();
};

#endif //TRAININGDATA_H
