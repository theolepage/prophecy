#pragma once

class Layer
{
public:
    Layer(nb_neurons)
        : nb_neurons_(nb_neurons)
    {}

    virtual void compile(Layer prev) = 0;

protected:
    bool compiled_;
    unsigned nb_neurons_;
};
