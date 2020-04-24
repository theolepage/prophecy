#pragma once

#include "../layer/hidden_layer.hh"
#include "../tensor/tensor.hh"
#include "../activation_function/activation_function.hh"

template <typename T>
class Conv2DLayer final : public HiddenLayer<T>
{
public:
    Conv2DLayer(int nb_neurons,
                const std::vector<int>& kernel_shape,
                int padding,
                int stride,
                const ActivationFunction<T>& activation)
    : HiddenLayer<T>(nb_neurons, activation)
    {
        kernel_shape_ = std::make_shared<std::vector<int>>(kernel_shape);
        padding_ = padding;
        stride_ = stride;
    }

    virtual ~Conv2DLayer() = default;

    Tensor<T> feedforward(const Tensor<T>& input, bool training)
    {
        int kernel_height = this->weights_->get_shape()[2];
        int kernel_width = this->weights_->get_shape()[3];
        int patch = this->weights_->get_shape()[1] * kernel_height * kernel_width;

        Tensor<T> weights_reshaped(this->weights_);
        weights_reshaped->reshape({ this->weights_->get_shape()[0], patch });
        auto input_reshaped = input.im2col(kernel_height, kernel_width, this->padding_, this->stride_);

        auto z = weights_reshaped.matmul(input_reshaped);
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
        this->weights_ = Tensor<T>({
                this->nb_neurons_,
                prev.lock()->get_nb_neurons(),
                this->kernel_shape_->at(0),
                this->kernel_shape_->at(1)
        });
        this->biases_ = Tensor<T>({ this->nb_neurons_ });
        this->weights_.fill(fill_type::RANDOM);
        this->biases_.fill(fill_type::ZEROS);

        // Initialize delta_weights_ and delta_biases_
        this->delta_weights_ = Tensor<T>(this->weights_->get_shape());
        this->delta_biases_ = Tensor<T>(this->biases_->get_shape());
        this->delta_weights_.fill(fill_type::ZEROS);
        this->delta_biases_.fill(fill_type::ZEROS);

        this->compiled_ = true;
        this->prev_ = prev;
        this->next_ = next;
    }

private:
    std::shared_ptr<std::vector<int>> kernel_shape_;
    int padding_;
    int stride_;
};
