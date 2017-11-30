#include <iostream>
#include <fstream>
#include <vector>
#include <armadillo>

#include "network.h"

using namespace arma;
using namespace std;

#define LEARNING_RATE 1
#define MIDDLE_LAYER 64

double trainWith(Network& network, mat& biasedSample, vec& results) {
    int correctNum = 0;
    for (int row = 0; row < 10; row++) {
        //cout << "training " << row << "/" << biasedSample.n_rows << endl;
        vec oneSample = biasedSample.row(row).t();
        network.compute(oneSample);
        if (network.update(results[row]))
            correctNum ++;
    }

    return 1.0f - (double)correctNum / 10;
}

int main() {
    mat trainSample;
    trainSample.load("TestDigitX.csv");

    vec bias(trainSample.n_rows, fill::ones);
    trainSample = arma::join_rows(trainSample, bias);

    vec results;
    results.load("TrainDigitY.csv");

    //cout << results << endl;
    
    Network network(LEARNING_RATE, MIDDLE_LAYER);

    for (int i = 0; i < 1000; i++) {
        double tError = trainWith(network, trainSample, results);
        cout << "Training Error: " << tError << endl;
    }


    return 0;
}