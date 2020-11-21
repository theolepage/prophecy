#pragma once

#include "activation_function/activation_function.hh"
#include "layer/processing_layer.hh"
#include "tensor/tensor.hh"

namespace prophecy
{
template <typename T = float>
class DenseLayer final : public ProcessingLayer<T>
{
  public:
    DenseLayer(const uint nb_neurons, const ActivationFunction<T>& activation)
        : ProcessingLayer<T>(nb_neurons, activation)
    {
    }

    virtual ~DenseLayer() = default;

    void compile(std::weak_ptr<Layer<T>> prev, std::shared_ptr<Layer<T>> next)
    {
        // Determine output shape
        std::vector<uint> out_shape = {this->nb_neurons_};
        this->out_shape_ = std::make_shared<std::vector<uint>>(out_shape);

        // Initialize weights and delta_weights
        const std::vector<uint> w_shape = {this->nb_neurons_,
                                           prev.lock()->get_out_shape()[0]};
        if (!this->compiled_ || this->weights_.get_shape() != w_shape)
        {
            this->weights_       = Tensor<T>(w_shape);
            this->delta_weights_ = Tensor<T>(w_shape);
        }
        this->weights_.fill(fill_type::RANDOM);
        this->delta_weights_.fill(fill_type::ZEROS);

        // Initialize biases and delta_biases
        const std::vector<uint> b_shape = {this->nb_neurons_, 1};
        if (!this->compiled_ || this->biases_.get_shape() != b_shape)
        {
            this->biases_       = Tensor<T>(b_shape);
            this->delta_biases_ = Tensor<T>(b_shape);
        }
        this->biases_.fill(fill_type::RANDOM);
        this->delta_biases_.fill(fill_type::ZEROS);

        this->compiled_ = true;
        this->prev_     = prev;
        this->next_     = next;
    }

    Tensor<T> feedforward(const Tensor<T>& input, const bool training)
    {
        auto z = this->weights_.matmul(input);
        z += this->biases_;
        auto a = z.map(this->activation_.get_f());

        if (training)
        {
            this->last_a_ = a;
            this->last_z_ = z;
        }

        if (this->next_ == nullptr)
            return a;
        return this->next_->feedforward(a, training);
    }

    void backpropagation(Tensor<T>& delta)
    {
        auto prev = this->prev_.lock();

        this->last_z_.map_inplace(this->activation_.get_fd());
        delta *= this->last_z_;

        // Compute db and dw
        this->delta_biases_ += delta;
        this->delta_weights_ += delta.matmul(prev->get_last_a().transpose());

        // Compute delta for previous layer and continue backpropagation
        delta = this->weights_.transpose().matmul(delta);
        prev->backpropagation(delta);
    }
};
} // namespace prophecy