#include "Network.h"
double Network::numberToAverage = 100.0;
Network::Network(const vector<unsigned>& topology)
{
	unsigned numLayers = topology.size();
	for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
		Layers.push_back(Layer());
		unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];


		for (unsigned neuronNum = 0; neuronNum < topology[layerNum]; ++neuronNum) {
			Layers.back().push_back(Neuron(numOutputs, neuronNum));
			//cout << "Neuron " << neuronNum << " from layer " << layerNum << " was born!" << endl;
		}
	}
}

void Network::feedForward(const vector<double>& inputValues)
{
	assert(inputValues.size() == Layers[0].size());
	for (unsigned i = 0; i < inputValues.size(); i++)
	{
		Layers[0][i].setOutputValue(inputValues[i]);
	}
	for (unsigned i = 1; i < Layers.size(); i++)
	{
		Layer& perviousLayer = Layers[i - 1];
		for (unsigned j = 0; j < Layers[i].size(); j++)
			Layers[i][j].feedForward(perviousLayer);
	}
}
void Network::printResult(ofstream& data)
{
	Layer& outputLayer = Layers.back();
	data << "Result: " << outputLayer[0].getOutputValue() << endl;
}

void Network::backPropagation(const vector<double>& targetValues)
{
	Layer& outputLayer = Layers.back();
	Error = 0.0;
	for (unsigned i = 0; i < outputLayer.size(); i++)
	{
		double delta = targetValues[i] - outputLayer[i].getOutputValue();
		Error = delta * delta;
	}
	Error /= outputLayer.size();
	Error = sqrt(Error);

	averageError = (averageError * numberToAverage + Error) / (numberToAverage + 1.0);

	for (unsigned i = 0; i < outputLayer.size(); i++)
	{
		outputLayer[i].calculateOutputGradients(targetValues[i]);
	}

	for (unsigned layerNumber = Layers.size() - 2; layerNumber > 0; layerNumber--)
	{
		Layer& hiddenLayer = Layers[layerNumber];
		Layer& nextLayer = Layers[layerNumber + 1];

		for (unsigned i = 0; i < hiddenLayer.size(); i++)
		{
			hiddenLayer[i].calculateHiddenGradients(nextLayer);
			cout << "Layer:" << layerNumber << "neuron:" << i << "gradient calculated." << endl;
		}
	}

	for (unsigned layerNumber = Layers.size() - 1; layerNumber > 0; layerNumber--)
	{
		Layer& layer = Layers[layerNumber];
		Layer& previousLayer = Layers[layerNumber - 1];

		for (unsigned i = 0; i < layer.size(); i++)
		{
			layer[i].updateWeights(previousLayer);
		}
	}

}
