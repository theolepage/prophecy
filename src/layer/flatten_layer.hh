#pragma once

#include "layer/layer.hh"
#include "tensor/tensor.hh"

template <typename T = float>
class FlattenLayer final : public Layer<T>
{
public:
    FlattenLayer()
        : Layer<T>()
    {}

    virtual ~FlattenLayer() = default;

    void compile(std::weak_ptr<Layer<T>> prev, std::shared_ptr<Layer<T>> next)
    {
        // Determine output shape
        unsigned int prev_size = 1;
        for (auto dim : prev.lock()->get_out_shape())
            prev_size *= dim;
        std::vector<unsigned int> out_shape({ prev_size, 1 });
        this->out_shape_ = std::make_shared<std::vector<unsigned int>>(out_shape);

        this->compiled_ = true;
        this->prev_ = prev;
        this->next_ = next;
    }

    Tensor<T> feedforward(const Tensor<T>& input, const bool training)
    {
        Tensor<T> out(input);
        out.reshape(*this->out_shape_);

        if (training)
            this->last_a_ = out;

        if (this->next_ == nullptr)
            return input;
        return this->next_->feedforward(out, training);
    }

    void backpropagation(Tensor<T>& delta)
    {
        auto prev = this->prev_.lock();

        prev->backpropagation(delta.reshape(prev->get_out_shape()));
    }
};