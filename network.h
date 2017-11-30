//
// Created by zixiong on 11/30/17.
//

#ifndef DIGITS_NETWORK_H
#define DIGITS_NETWORK_H

#include <armadillo>

#define DIM_IN 785
#define DIM_OUT 10

using namespace std;
using namespace arma;

class Network {
public:
    /* Constructor */
    Network(double eta, int countMiddleLayer);

    /* Takes the input vector (bias included, gives the output of the final layer */
    vec& compute(vec& input);

    /* Takes the correct answer, using it and the last computing result, update weights. Returns wether predicted correctly */
    bool update(int i);

    /* argmax of the lasts compute */
    int result();

private:
    vec layer0Val;
    vec layer1Val;
    vec layer1Sig;
    vec layer1Del;
    vec layer2Val;
    vec layer2Sig;
    vec layer2Del;

    mat weights12;
    mat weights23;

    double _eta;

    int _countMiddleLayer;

    inline static vec sigmoid(vec& v) {
        vec ret(v.size());
        for (int i = 0; i < v.size(); i++)
            ret[i] = 1.0f / (1.0f + exp(-v[i]));
        //return 1.0f / (1.0f + arma::exp(-v));
        return ret;
    }

    inline static vec derivative(vec& v) {
        vec one = arma::ones(v.size());
        return sigmoid(v) % (one - sigmoid(v));
    }
};


#endif //DIGITS_NETWORK_H
