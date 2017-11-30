//
// Created by zixiong on 11/30/17.
//

#include "network.h"

#include <cassert>

Network::Network(double eta, int countMiddleLayer) :
    _eta(eta),
    _countMiddleLayer(countMiddleLayer),
    layer0Val(DIM_IN),
    layer1Val(countMiddleLayer),
    layer1Sig(countMiddleLayer),
    layer1Del(countMiddleLayer),
    layer2Val(DIM_OUT),
    layer2Sig(DIM_OUT),
    layer2Del(DIM_OUT),
    weights12(countMiddleLayer, DIM_IN),
    weights23(DIM_OUT, countMiddleLayer + 1)
{
    /*randn() uses a normal/Gaussian distribution with zero mean and unit variance*/

    weights12.fill(0);
    weights23.fill(0);
}


vec& Network::compute(vec& input) {
    assert(input.n_rows == DIM_IN);
    
    layer0Val = input;
    layer1Val = weights12 * input;
    layer1Sig = sigmoid(layer1Val);
    vec layer1SigPlusBias = layer1Sig;
    layer1SigPlusBias.insert_rows(_countMiddleLayer, 1);
    layer1SigPlusBias[_countMiddleLayer] = 1;

    layer2Val = weights23 * layer1SigPlusBias;
    layer2Sig = sigmoid(layer2Val);

    return layer2Sig;
}

bool Network::update(int n) {
    double max = layer2Sig.max();
    cout << layer2Sig << endl;
    bool ret = (max == layer2Sig[n]);

    vec y(DIM_OUT, fill::zeros);
    y[n] = 1.0f;


    double loss = arma::norm(layer2Sig - y);
    //cout << "loss: " << loss << endl;
    

    // update the last layer
    vec layer1SigPlusBias = layer1Sig;
    layer1SigPlusBias.insert_rows(_countMiddleLayer, 1);
    layer1SigPlusBias[_countMiddleLayer] = 1;

    layer2Del = (layer2Sig - y) % derivative(layer2Val);
    mat delta1 = _eta * (layer2Del * layer1SigPlusBias.t());

    // update the second to last layer
    vec sums = weights23.t() * layer2Del;
    //cout << sums.n_rows << endl;
    sums.shed_row(_countMiddleLayer);
    layer1Del = derivative(layer1Val) % sums;

    mat delta2 = _eta * (layer1Del * layer0Val.t());
    weights12 -= delta2;
    //cout << layer1Del << endl;
    weights23 -= delta1;
    return ret;
}