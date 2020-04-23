#pragma once

#include "layer.hh"
#include "../tensor/tensor.hh"

template <typename T>
class InputLayer final : public Layer<T>
{
public:
    InputLayer(int nb_neurons)
        : Layer<T>(nb_neurons)
    {}

    virtual ~InputLayer() = default;

    Tensor<T> feedforward(const Tensor<T>& input, bool training)
    {
        this->last_a_ = input;
        return this->next_->feedforward(input, training);
    }

    void backpropagation(const Tensor<T>* const)
    {
        return;
    }
};
