#pragma once

#include "../layer/processing_layer.hh"
#include "../tensor/tensor.hh"
#include "../activation_function/activation_function.hh"

template <typename T>
class Conv2DLayer final : public ProcessingLayer<T>
{
public:
    Conv2DLayer(int nb_filters,
                const std::vector<int>& kernel_shape,
                int padding,
                int stride,
                const ActivationFunction<T>& activation)
        : ProcessingLayer<T>(nb_filters, activation)
        , kernel_shape_(std::make_shared<std::vector<int>>(kernel_shape))
        , padding_(padding)
        , stride_(stride)
    {}

    Conv2DLayer(int nb_filters,
                const std::vector<int>& kernel_shape,
                const ActivationFunction<T>& activation)
        : ProcessingLayer<T>(nb_filters, activation)
        , kernel_shape_(std::make_shared<std::vector<int>>(kernel_shape))
        , padding_(0)
        , stride_(1)
    {}

    virtual ~Conv2DLayer() = default;

    void compile(std::weak_ptr<Layer<T>> prev, std::shared_ptr<Layer<T>> next)
    {
        // Determine output shape
        auto prev_shape = prev.lock()->get_out_shape();
        int c = this->nb_neurons_;
        int h = 1 + (prev_shape[1] + 2 * this->padding_ - this->kernel_shape_->at(0)) / this->stride_;
        int w = 1 + (prev_shape[2] + 2 * this->padding_ - this->kernel_shape_->at(1)) / this->stride_;
        std::vector<int> out_shape({ c, h, w });
        this->out_shape_ = std::make_shared<std::vector<int>>(out_shape);

        // Initialize weights and biases
        this->weights_ = Tensor<T>({
            this->nb_neurons_,
            prev_shape[0],
            this->kernel_shape_->at(0),
            this->kernel_shape_->at(1)
        });
        this->biases_ = Tensor<T>({ this->nb_neurons_ });
        this->weights_.fill(fill_type::RANDOM);
        this->biases_.fill(fill_type::ZEROS);

        // Initialize delta_weights_ and delta_biases_
        this->delta_weights_ = Tensor<T>(this->weights_.get_shape());
        this->delta_biases_ = Tensor<T>(this->biases_.get_shape());
        this->delta_weights_.fill(fill_type::ZEROS);
        this->delta_biases_.fill(fill_type::ZEROS);

        this->compiled_ = true;
        this->prev_ = prev;
        this->next_ = next;
    }

    Tensor<T> feedforward(const Tensor<T>& input, bool training)
    {
        int kernel_channels = this->weights_.get_shape()[1];
        int kernel_height = this->weights_.get_shape()[2];
        int kernel_width = this->weights_.get_shape()[3];

        Tensor<T> weights_reshaped(this->weights_);
        weights_reshaped.reshape({
            this->nb_neurons_,
            kernel_channels * kernel_height * kernel_width
        });

        Tensor<T> in(input);
        auto input_reshaped = in.im2col(kernel_height, kernel_width, this->padding_, this->stride_);

        auto z = weights_reshaped.matmul(input_reshaped);
        z.reshape(*this->out_shape_);

        for (int i = 0; i < this->nb_neurons_; i++)
            z.extract({ i }) += this->biases_({ i });
        auto a = z.map(this->activation_.f_);

        if (training)
        {
            this->last_a_ = a;
            this->last_z_ = z;
            this->last_a_col_ = input_reshaped;
        }

        if (this->next_ == nullptr)
            return a;
        return this->next_->feedforward(a, training);
    }

    void backpropagation(Tensor<T>& delta)
    {
        auto prev = this->prev_.lock();
        
        this->last_z_.map_inplace(this->activation_.fd_);
        delta *= this->last_z_;

        // Compute db
        this->delta_biases_ += delta.sum({ 1, 2 });

        // Compute dw
        Tensor<T> delta_reshaped(delta);
        delta_reshaped.reshape({
                delta.get_shape()[0],
                delta.get_shape()[1] * delta.get_shape()[2]
        });
        auto dw = delta_reshaped.matmul(last_a_col_.transpose());
        this->delta_weights_ += dw.reshape(this->weights_.get_shape());

        // Compute delta
        int kernel_channels = this->weights_.get_shape()[1];
        int kernel_height = this->weights_.get_shape()[2];
        int kernel_width = this->weights_.get_shape()[3];
        Tensor<T> weights_reshaped(this->weights_);
        weights_reshaped.reshape({
            this->nb_neurons_,
            kernel_channels * kernel_height * kernel_width
        });

        auto delta_col = weights_reshaped.transpose().matmul(delta_reshaped);
        delta = delta_col.col2im(prev->get_out_shape(), kernel_height, kernel_width, this->padding_, this->stride_);
        
        prev->backpropagation(delta);
    }

private:
    std::shared_ptr<std::vector<int>> kernel_shape_;
    int padding_;
    int stride_;

    Tensor<T> last_a_col_;

    int dim_after_conv(int dim) const
    {
        return 1 + (dim + 2 * this->padding_ - this->shape_->at(2)) / this->stride_;
    }
};
