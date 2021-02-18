#pragma once

#include "activation_function/activation_function.hh"
#include "layer/processing_layer.hh"
#include "xtensor-blas/xlinalg.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xvectorize.hpp"

namespace prophecy
{
template <typename T = float>
class DenseLayer final : public ProcessingLayer<T>
{
  public:
    explicit DenseLayer(
        const uint                                    nb_neurons,
        const std::shared_ptr<ActivationFunction<T>>& activation,
        const std::string&                            init="xavier");

    virtual ~DenseLayer() = default;

    void compile(std::weak_ptr<Layer<T>> prev, std::shared_ptr<Layer<T>> next);

    xt::xarray<T> feedforward(const xt::xarray<T>& input, const bool training);

    void backpropagation(xt::xarray<T>& delta);

    std::string get_name() const;

  private:
    std::string init_;
};

template <typename T>
DenseLayer<T>::DenseLayer(
    const uint                                    nb_neurons,
    const std::shared_ptr<ActivationFunction<T>>& activation,
    const std::string&                            init)
    : ProcessingLayer<T>(nb_neurons, activation)
    , init_(init)
{
}

template <typename T>
void DenseLayer<T>::compile(std::weak_ptr<Layer<T>>   prev,
                            std::shared_ptr<Layer<T>> next)
{
    // Determine output shape
    std::vector<uint> out_shape = {this->nb_neurons_};
    this->out_shape_ = std::make_shared<std::vector<uint>>(out_shape);

    // Initialize weights and delta_weights
    const std::vector<uint> w_shape = {this->nb_neurons_,
                                       prev.lock()->get_out_shape()[0]};
    if (init_ == "xavier")
    {
        const T weights_std =
            sqrt(static_cast<T>(2) / (w_shape.front() + w_shape.back()));
        this->weights_ = xt::random::randn<T>(w_shape, 0, weights_std);
    }
    else
    {
        this->weights_ = xt::random::randn<T>(w_shape, 0, 1);
    }
    this->delta_weights_ = xt::zeros<T>(w_shape);

    // Initialize biases and delta_biases
    const std::vector<uint> b_shape = {this->nb_neurons_};
    this->delta_biases_             = xt::zeros<T>(b_shape);

    if (init_ == "xavier")
        this->biases_ = xt::zeros<T>(b_shape);
    else
        this->biases_ = xt::random::randn<T>(b_shape, 0, 1);

    this->compiled_ = true;
    this->prev_     = prev;
    this->next_     = next;
}

template <typename T>
xt::xarray<T> DenseLayer<T>::feedforward(const xt::xarray<T>& input,
                                         const bool           training)
{
    auto z = xt::linalg::dot(input, xt::transpose(this->weights_));
    z += this->biases_;

    auto f = xt::vectorize(this->activation_->get_f());
    auto a = f(z);

    if (training)
    {
        this->last_a_ = a;
        this->last_z_ = z;
    }

    if (this->next_ == nullptr)
        return a;
    return this->next_->feedforward(a, training);
}

template <typename T>
void DenseLayer<T>::backpropagation(xt::xarray<T>& delta)
{
    auto prev = this->prev_.lock();

    auto fd = xt::vectorize(this->activation_->get_fd());

    this->last_z_ = fd(this->last_z_);
    delta *= this->last_z_;

    // Compute db and dw
    this->delta_biases_ += xt::sum(delta, 0);
    this->delta_weights_ +=
        xt::linalg::dot(xt::transpose(delta), prev->get_last_a());

    // Compute delta for previous layer and continue backpropagation
    delta = xt::linalg::dot(delta, this->weights_);
    prev->backpropagation(delta);
}

template <typename T>
std::string DenseLayer<T>::get_name() const
{
    return "Dense";
}
} // namespace prophecy
