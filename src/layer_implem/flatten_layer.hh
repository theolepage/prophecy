#pragma once

#include "../layer/layer.hh"
#include "../tensor/tensor.hh"

template <typename T>
class FlattenLayer final : public Layer<T>
{
public:
    FlattenLayer(const std::vector<int>& shape)
        : Layer<T>(shape)
    {}

    virtual ~FlattenLayer() = default;

    Tensor<T> feedforward(const Tensor<T>& input, bool training)
    {
        Tensor<T> out(input);
        out.reshape(*this->shape_);

        if (training)
        {
            this->last_a_ = out;
            this->previous_shape_ = std::make_shared<std::vector<int>>(input.get_shape());
        }

        if (this->next_ == nullptr)
            return input;
        return this->next_->feedforward(out, training);
    }

    void backpropagation(Tensor<T>& delta)
    {
        auto prev = this->prev_.lock();

        prev->backpropagation(delta.reshape(*this->previous_shape_));
    }

private:
    std::shared_ptr<std::vector<int>> previous_shape_;
};