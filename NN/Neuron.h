#pragma once
#include <iostream>
#include <time.h>
#include <vector>

using namespace std;
class Neuron;
typedef vector<Neuron> Layer;
struct Connection
{
    double Weight;
    double deltaWeight;
};

class Neuron
{
public:
    Neuron(unsigned numberOfOutputs, unsigned neuronIndex);
    void setOutputValue(const double& inputValue) { outputValue = inputValue; }
    void feedForward(const Layer& previousLayer);
    double getOutputValue() const { return outputValue; }
    double getOutputWeightof(unsigned n) const { return outputWeights[n].Weight; }
    void calculateOutputGradients(double targetVAlue);
    void calculateHiddenGradients(const Layer& nextLayer);
    void updateWeights(Layer& pervLayer);
private:
    static double traningRate;
    static double Momentum;
    double Gradient;
    static double activationFunction(double sum);
    static double activationFunctionDerivative(double sum);
    double randomWeight() { return rand() / double(RAND_MAX); }
    double sumDOW(const Layer& nextLayer)const;
    unsigned Index;
    double outputValue;
    vector<Connection> outputWeights;
    double bias;
};

