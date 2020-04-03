#pragma once

#include "layer.hh"

class InputLayer : public Layer
{
public:
    InputLayer(double nb_neurons)
        : Layer(nb_neurons)
    {}
};
