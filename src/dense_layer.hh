#pragma once

#include <functional>

#include "layer.hh"
#include "matrix.hh"
#include "activation_function.hh"

class DenseLayer : public Layer
{
public:
    DenseLayer(unsigned nb_neurons, ActivationFunction activation)
        : Layer(nb_neurons)
        , activation_(activation)
    {}

private:
    Matrix weights_;
    Matrix biases_;
    ActivationFunction activation_;
};
