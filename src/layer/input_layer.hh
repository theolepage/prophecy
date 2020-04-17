#pragma once

#include "layer.hh"
#include "../matrix/matrix.hh"

template <typename T>
class InputLayer : public Layer<T>
{
public:
    InputLayer(int nb_neurons)
        : Layer<T>(nb_neurons)
    {}

    virtual Matrix<T> feedforward(const Matrix<T>& input, bool training)
    {
        this->last_a_ = input;
        return this->next_->feedforward(input, training);
    }

    void backpropagation(const Matrix<T>* const)
    {
        return;
    }
};
