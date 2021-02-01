#pragma once

#include "layer.hh"

namespace prophecy
{
template <typename T = float>
class InputLayer final : public Layer<T>
{
  public:
    explicit InputLayer(const std::vector<uint>& out_shape);

    virtual ~InputLayer() = default;

    xt::xarray<T> feedforward(const xt::xarray<T>& input, bool training);

    void backpropagation(xt::xarray<T>&);
};

template <typename T>
InputLayer<T>::InputLayer(const std::vector<uint>& out_shape)
    : Layer<T>(out_shape)
{
}

template <typename T>
xt::xarray<T> InputLayer<T>::feedforward(const xt::xarray<T>& input,
                                         bool                 training)
{
    this->last_a_ = input;
    return this->next_->feedforward(input, training);
}

template <typename T>
void InputLayer<T>::backpropagation(xt::xarray<T>&)
{
    return;
}
} // namespace prophecy