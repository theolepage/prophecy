#pragma once

#include <memory>
#include <functional>

#include "layer.hh"
#include "../tensor/tensor.hh"
#include "../activation_function/activation_function.hh"

template <typename T>
class ProcessingLayer : public Layer<T>
{
public:

    ProcessingLayer(const std::vector<int>& shape, const ActivationFunction<T>& activation)
        : Layer<T>(shape)
        , activation_(activation)
    {}

    virtual ~ProcessingLayer() = default;

    virtual void update(double learning_rate)
    {
         // Update weights_ and biases_
        this->delta_weights_ *= learning_rate;
        this->weights_ -=  this->delta_weights_;
        this->delta_biases_ *= learning_rate;
        this->biases_ -= this->delta_biases_;

        // Reset delta_weights_ and delta_biases_
        this->delta_weights_.fill(fill_type::ZEROS);
        this->delta_biases_.fill(fill_type::ZEROS);
    }

protected:
    Tensor<T> weights_;
    Tensor<T> biases_;

    Tensor<T> delta_weights_;
    Tensor<T> delta_biases_;

    ActivationFunction<T> activation_;
};