#pragma once

#include "matrix.hh"

class Layer
{
public:
    Layer(unsigned nb_neurons);

    unsigned get_nb_neurons();

protected:
    unsigned nb_neurons_;
};
