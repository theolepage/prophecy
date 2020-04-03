#pragma once

#include <functional>

#include "layer.hh"
#include "matrix.hh"

class DenseLayer : public Layer
{
public:
    DenseLayer(nb_neurons, activation)
        : Layer(nb_neurons)
        , activation_(activation)
    {}

private:
    Matrix weights_;
    Matrix biases_;
    std::function<double(double)> activation_;
};
