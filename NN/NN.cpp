#include <iostream>
#include <time.h>
#include <vector>
#include<fstream>
#include "Network.h"
#include "Neuron.h"
using namespace std;

int main()
{
    srand((unsigned)time(NULL));
    int n1, n2, t;
    ofstream data ("Text.txt");
    vector<double> inputValues;
    vector<double> targetValues;

    vector<unsigned> topology;// how many layers do we have in our nn and in each layer how many neurons 
    topology.push_back(2);//input layer
    topology.push_back(4);//hidden layer
    topology.push_back(1);//output layer
    Network mynetwork(topology);


    for (int i = 0; i <= 4000; i++)
    {
        data << endl << "Pass: " << i << endl;
        n1 = (int)(2.0 * rand() / double(RAND_MAX));//or 0 or 1
        n2 = (int)(2.0 * rand() / double(RAND_MAX));//or 0 or 1
        //n3 = (int)(2.0 * rand() / double(RAND_MAX));
        //n4 = (int)(2.0 * rand() / double(RAND_MAX));
        t = n1 ^ n2;//n3 ^ n4;
        inputValues.clear();
        targetValues.clear();
        inputValues.push_back(n1);
        inputValues.push_back(n2);
        //inputValues.push_back(n3);
        //inputValues.push_back(n4);
        targetValues.push_back(t);
        data << "Inputs: " << n1 << " " << n2 << " " << endl;
        //    << n3 << " " << n4 << endl;
        mynetwork.feedForward(inputValues);
        mynetwork.printResult(data);
        data << "Target: " << t << endl;
        data << "How well am I working : " << mynetwork.getAvarageError() << endl;
        mynetwork.backPropagation(targetValues);
    }
    
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
