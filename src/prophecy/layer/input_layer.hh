#pragma once

#include "layer.hh"
#include "tensor/tensor.hh"

namespace prophecy
{
template <typename T = float>
class InputLayer final : public Layer<T>
{
  public:
    explicit InputLayer(const std::vector<uint>& out_shape);

    virtual ~InputLayer() = default;

    Tensor<T> feedforward(const Tensor<T>& input, bool training);

    void backpropagation(Tensor<T>&);
};

template <typename T>
InputLayer<T>::InputLayer(const std::vector<uint>& out_shape)
    : Layer<T>(out_shape)
{
}

template <typename T>
Tensor<T> InputLayer<T>::feedforward(const Tensor<T>& input, bool training)
{
    this->last_a_ = input;
    return this->next_->feedforward(input, training);
}

template <typename T>
void InputLayer<T>::backpropagation(Tensor<T>&)
{
    return;
}
} // namespace prophecy