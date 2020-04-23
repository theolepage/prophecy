#pragma once

#include "../layer/hidden_layer.hh"
#include "../tensor/tensor.hh"
#include "../activation_function/activation_function.hh"

template <typename T>
class DenseLayer final : public HiddenLayer<T>
{
public:
    DenseLayer(int nb_neurons,
               const ActivationFunction<T>& activation)
    : HiddenLayer<T>(nb_neurons, activation)
    {}

    virtual ~DenseLayer() = default;

    Tensor<T> feedforward(const Tensor<T>& input, bool training)
    {
        auto z = this->weights_.matmul(input);
        z += this->biases_;
        auto a = z.map(this->activation_.f_);

        if (training)
        {
            this->last_a_ = a;
            this->last_z_ = z;
        }

        if (this->next_ == nullptr)
            return a;
        return this->next_->feedforward(a, training);
    }

    void backpropagation(const Tensor<T>* const y)
    {
        if (this->prev_.expired()) // If we reach InputLayer
            return;

        auto next = std::dynamic_pointer_cast<HiddenLayer<T>>(this->next_);

        this->last_z_.map_inplace(this->activation_.fd_); // Avoid creating a new Tensor below
        if (y != nullptr)
        {
            this->last_a_ -= *y; // Same
            this->last_a_ *= this->last_z_; // Same
            this->delta_ = this->last_a_;
        }
        else
        {
            this->last_z_ *= next->get_weights().transpose().matmul(next->get_delta()); // Same
            this->delta_ = this->last_z_;
        }

        this->delta_biases_ += this->delta_;
        this->delta_weights_ += this->delta_.matmul(this->prev_.lock()->get_last_a().transpose());
        this->prev_.lock()->backpropagation(nullptr);
    }

    void update(T learning_rate)
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

    void compile(std::weak_ptr<Layer<T>> prev,
                 std::shared_ptr<Layer<T>> next)
    {
        // Initialize weights and biases
        this->weights_ = Tensor<T>({this->nb_neurons_, prev.lock()->get_nb_neurons()});
        this->biases_ = Tensor<T>({this->nb_neurons_, 1});
        this->weights_.fill(fill_type::RANDOM);
        this->biases_.fill(fill_type::RANDOM);

        // Initialize delta_weights_ and delta_biases_
        this->delta_weights_ = Tensor<T>({this->nb_neurons_, prev.lock()->get_nb_neurons()});
        this->delta_biases_ = Tensor<T>({this->nb_neurons_, 1});
        this->delta_weights_.fill(fill_type::ZEROS);
        this->delta_biases_.fill(fill_type::ZEROS);

        this->compiled_ = true;
        this->prev_ = prev;
        this->next_ = next;
    }

};
