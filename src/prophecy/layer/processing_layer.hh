#pragma once

#include <functional>
#include <memory>

#include "activation_function/activation_function.hh"
#include "layer.hh"

namespace prophecy
{
template <typename T = float>
class ProcessingLayer : public Layer<T>
{
  public:
    explicit ProcessingLayer(
        const uint                                    nb_neurons,
        const std::shared_ptr<ActivationFunction<T>>& activation);

    virtual ~ProcessingLayer() = default;

    virtual void update(const double learning_rate, const uint batch_size);

    virtual uint get_params_count() const;

  protected:
    const uint nb_neurons_;

    xt::xarray<T> weights_;
    xt::xarray<T> biases_;

    xt::xarray<T> delta_weights_;
    xt::xarray<T> delta_biases_;

    const std::shared_ptr<ActivationFunction<T>> activation_;
};

template <typename T>
ProcessingLayer<T>::ProcessingLayer(
    const uint                                    nb_neurons,
    const std::shared_ptr<ActivationFunction<T>>& activation)
    : Layer<T>()
    , nb_neurons_(nb_neurons)
    , activation_(activation)
{
}

template <typename T>
void ProcessingLayer<T>::update(const double learning_rate,
                                const uint   batch_size)
{
    // Update weights and biases
    weights_ -= (learning_rate / batch_size) * delta_weights_;
    biases_ -= (learning_rate / batch_size) * delta_biases_;

    // Reset delta weights and delta biases
    delta_weights_ = xt::zeros<T>(weights_.shape());
    delta_biases_  = xt::zeros<T>(biases_.shape());
}

template <typename T>
uint ProcessingLayer<T>::get_params_count() const
{
    return this->weights_.size() + this->biases_.size();
}
} // namespace prophecy