//
// Created by shane on 6/23/2025.
//

#ifndef ACTIVATIONFUNCTION_H
#define ACTIVATIONFUNCTION_H
#include <valarray>


class ActivationFunction {
public:
    virtual double calculate(double z) = 0;
    virtual double calculateDerivative(double z) = 0;

    virtual ~ActivationFunction() = default;
};

class Sigmoid final : public ActivationFunction {
public:
    double calculate(const double z) override {
        return 1 / (1 + exp(-z));
    }
    double calculateDerivative(const double z) override {
        return calculate(z) * (1 - calculate(z));
    }
};

class Tanh : public ActivationFunction {
    double calculate(double z) override {
        return tanh(z);
    }
    double calculateDerivative(double z) override {
        return (1 - pow(calculate(z), 2));
    }
};

class ReLU : public ActivationFunction {
public:
    double calculate(double z) override {
        return z < 0 ? 0 : z;
    }
    double calculateDerivative(double z) override {
        return z < 0 ? 0 : 1;
    }
};
#endif //ACTIVATIONFUNCTION_H
