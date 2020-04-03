#pragma once

#include "matrix.hh"

class Layer
{
public:
    Layer(unsigned nb_neurons)
        : nb_neurons_(nb_neurons)
    {}

    unsigned get_nb_neurons()
    {
        return nb_neurons_;
    }

protected:
    unsigned nb_neurons_;
};
