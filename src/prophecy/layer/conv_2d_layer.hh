#pragma once

#include "activation_function/activation_function.hh"
#include "layer/processing_layer.hh"
#include "tensor/tensor.hh"

namespace prophecy
{
template <typename T = float>
class Conv2DLayer final : public ProcessingLayer<T>
{
  public:
    Conv2DLayer(uint                         nb_filters,
                const std::vector<uint>&     kernel_shape,
                const uint                   padding,
                const uint                   stride,
                const ActivationFunction<T>& activation)
        : ProcessingLayer<T>(nb_filters, activation),
          kernel_shape_(std::make_shared<std::vector<uint>>(kernel_shape)),
          padding_(padding), stride_(stride)
    {
    }

    Conv2DLayer(const uint                   nb_filters,
                const std::vector<uint>&     kernel_shape,
                const ActivationFunction<T>& activation)
        : ProcessingLayer<T>(nb_filters, activation),
          kernel_shape_(std::make_shared<std::vector<uint>>(kernel_shape)),
          padding_(0), stride_(1)
    {
    }

    virtual ~Conv2DLayer() = default;

    void compile(std::weak_ptr<Layer<T>> prev, std::shared_ptr<Layer<T>> next)
    {
        // Determine output shape
        auto       prev_shape = prev.lock()->get_out_shape();
        const uint c          = this->nb_neurons_;
        const uint h          = 1 + (prev_shape[1] + 2 * this->padding_ -
                            this->kernel_shape_->at(0)) /
                               this->stride_;
        const uint w = 1 + (prev_shape[2] + 2 * this->padding_ -
                            this->kernel_shape_->at(1)) /
                               this->stride_;
        std::vector<uint> out_shape({c, h, w});
        this->out_shape_ = std::make_shared<std::vector<uint>>(out_shape);

        // Initialize weights and delta_weights
        const std::vector<uint> w_shape = {this->nb_neurons_,
                                           prev_shape[0],
                                           this->kernel_shape_->at(0),
                                           this->kernel_shape_->at(1)};
        if (!this->compiled_ || this->weights_.get_shape() != w_shape)
        {
            this->weights_       = Tensor<T>(w_shape);
            this->delta_weights_ = Tensor<T>(w_shape);
        }
        this->weights_.fill(fill_type::RANDOM);
        this->delta_weights_.fill(fill_type::ZEROS);

        // Initialize biases and delta_biases
        const std::vector<uint> b_shape = {this->nb_neurons_};
        if (!this->compiled_ || this->biases_.get_shape() != b_shape)
        {
            this->biases_       = Tensor<T>(b_shape);
            this->delta_biases_ = Tensor<T>(b_shape);
        }
        this->biases_.fill(fill_type::ZEROS);
        this->delta_biases_.fill(fill_type::ZEROS);

        this->compiled_ = true;
        this->prev_     = prev;
        this->next_     = next;
    }

    Tensor<T> feedforward(const Tensor<T>& input, bool training)
    {
        const uint kernel_channels = this->weights_.get_shape()[1];
        const uint kernel_height   = this->weights_.get_shape()[2];
        const uint kernel_width    = this->weights_.get_shape()[3];

        Tensor<T> weights_reshaped(this->weights_);
        weights_reshaped.reshape(
            {this->nb_neurons_,
             kernel_channels * kernel_height * kernel_width});

        Tensor<T> in(input);
        auto      input_reshaped = in.im2col(
            kernel_height, kernel_width, this->padding_, this->stride_);

        auto z = weights_reshaped.matmul(input_reshaped);
        z.reshape(*this->out_shape_);

        for (uint i = 0; i < this->nb_neurons_; i++)
            z.extract({i}) += this->biases_({i});
        auto a = z.map(this->activation_.get_f());

        if (training)
        {
            this->last_a_     = a;
            this->last_z_     = z;
            this->last_a_col_ = input_reshaped;
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

        // Compute db
        this->delta_biases_ += delta.sum({1, 2});

        // Compute dw
        Tensor<T> delta_reshaped(delta);
        delta_reshaped.reshape({delta.get_shape()[0],
                                delta.get_shape()[1] * delta.get_shape()[2]});
        auto dw = delta_reshaped.matmul(last_a_col_.transpose());
        this->delta_weights_ += dw.reshape(this->weights_.get_shape());

        // Compute delta
        const uint kernel_channels = this->weights_.get_shape()[1];
        const uint kernel_height   = this->weights_.get_shape()[2];
        const uint kernel_width    = this->weights_.get_shape()[3];
        Tensor<T>  weights_reshaped(this->weights_);
        weights_reshaped.reshape(
            {this->nb_neurons_,
             kernel_channels * kernel_height * kernel_width});

        auto delta_col = weights_reshaped.transpose().matmul(delta_reshaped);
        delta          = delta_col.col2im(prev->get_out_shape(),
                                 kernel_height,
                                 kernel_width,
                                 this->padding_,
                                 this->stride_);

        prev->backpropagation(delta);
    }

  private:
    std::shared_ptr<std::vector<uint>> kernel_shape_;
    const uint                         padding_;
    const uint                         stride_;

    Tensor<T> last_a_col_;

    uint dim_after_conv(const uint dim) const
    {
        return 1 +
               (dim + 2 * this->padding_ - this->shape_->at(2)) / this->stride_;
    }
};
} // namespace prophecy