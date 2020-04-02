#pragma once

#include "layer.hh"
#include "matrix.hh"

class DenseLayer : public Layer
{
public:
    DenseLayer(nb_neurons)
        : Layer(nb_neurons)
    {}

private:
    Matrix weights_;
    Matrix biases_;
};
