#pragma once

#include "../layer/layer.hh"
#include "../tensor/tensor.hh"

template <typename T>
class FlattenLayer final : public Layer<T>
{
public:
    FlattenLayer()
        : Layer<T>({ 0 })
    {}

    virtual ~FlattenLayer() = default;

    Tensor<T> feedforward(const Tensor<T>& input, bool training)
    {
        this->shape_ = std::make_shared<std::vector<int>>({ input.get_size() });

        Tensor<T> out(input);
        out.reshape(*this->shape_);

        if (training)
            this->last_a_ = out;

        if (this->next_ == nullptr)
            return input;
        return this->next_->feedforward(input, training);
    }

    void backpropagation(Tensor<T>& delta)
    {
        auto prev = this->prev_.lock();

        prev->backpropagation(delta.reshape(prev->get_shape()));
    }
};