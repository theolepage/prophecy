#pragma once

#include <memory>
#include <functional>

#include "layer.hh"
#include "../tensor/tensor.hh"
#include "../activation_function/activation_function.hh"

template <typename T>
class HiddenLayer : public Layer<T>
{
public:

    HiddenLayer(const std::vector<int>& shape, const ActivationFunction<T>& activation)
        : Layer<T>(shape)
        , activation_(activation)
    {}

    virtual ~HiddenLayer() = default;

    virtual void update(T learning_rate) = 0;

    Tensor<T>& get_weights(void) { return weights_; };

    Tensor<T>& get_biases(void) { return biases_; };

    Tensor<T>& get_delta(void) { return delta_; }

protected:
    Tensor<T> weights_;
    Tensor<T> biases_;

    Tensor<T> delta_;
    Tensor<T> delta_weights_;
    Tensor<T> delta_biases_;

    ActivationFunction<T> activation_;
};