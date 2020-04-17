#pragma once

#include <memory>
#include <functional>

#include "layer.hh"
#include "../matrix/matrix.hh"
#include "../activation_function/activation_function.hh"

template <typename T>
class HiddenLayer : public Layer<T>
{
public:
    HiddenLayer(int nb_neurons, const ActivationFunction<T>& activation) : Layer<T>(nb_neurons)
    , activation_(activation)
    {}

    virtual void update(T learning_rate) = 0;

    Matrix<T>& get_weights(void) { return weights_; };

protected:
    Matrix<T> weights_;
    Matrix<T> biases_;

    Matrix<T> delta_weights_;
    Matrix<T> delta_biases_;

    ActivationFunction<T> activation_;
};
