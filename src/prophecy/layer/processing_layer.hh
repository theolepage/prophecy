#pragma once

#include <functional>
#include <memory>

#include "activation_function/activation_function.hh"
#include "layer.hh"
#include "tensor/tensor.hh"

namespace prophecy
{
template <typename T = float>
class ProcessingLayer : public Layer<T>
{
  public:
    ProcessingLayer(const uint                   nb_neurons,
                    const ActivationFunction<T>& activation)
        : Layer<T>(), nb_neurons_(nb_neurons), activation_(activation)
    {
    }

    virtual ~ProcessingLayer() = default;

    virtual void update(double learning_rate)
    {
        // Update weights_ and biases_
        this->delta_weights_ *= learning_rate;
        this->weights_ -= this->delta_weights_;
        this->delta_biases_ *= learning_rate;
        this->biases_ -= this->delta_biases_;

        // Reset delta_weights_ and delta_biases_
        this->delta_weights_.fill(fill_type::ZEROS);
        this->delta_biases_.fill(fill_type::ZEROS);
    }

  protected:
    const uint nb_neurons_;

    Tensor<T> weights_;
    Tensor<T> biases_;

    Tensor<T> delta_weights_;
    Tensor<T> delta_biases_;

    const ActivationFunction<T>& activation_;
};
} // namespace prophecy