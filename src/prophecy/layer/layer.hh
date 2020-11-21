#pragma once

#include <memory>

#include "tensor/tensor.hh"

namespace prophecy
{
template <typename T = float>
class Layer
{
  public:
    Layer(const std::vector<uint>& out_shape)
        : out_shape_(std::make_shared<std::vector<uint>>(out_shape))
    {
    }

    Layer() : out_shape_(nullptr) {}

    virtual ~Layer() = default;

    virtual void compile(std::weak_ptr<Layer<T>>   prev,
                         std::shared_ptr<Layer<T>> next)
    {
        this->compiled_ = true;
        this->prev_     = prev;
        this->next_     = next;
    }

    virtual Tensor<T> feedforward(const Tensor<T>& input,
                                  const bool       training) = 0;

    virtual void backpropagation(Tensor<T>& delta) = 0;

    virtual Tensor<T>& cost(const Tensor<T>& y)
    {
        this->last_a_ -= y;
        return this->last_a_;
    }

    std::vector<uint> get_out_shape() const { return *this->out_shape_; }

    Tensor<T>& get_last_a() { return this->last_a_; }

    Tensor<T>& get_last_z() { return this->last_z_; }

  protected:
    bool                               compiled_;
    std::shared_ptr<std::vector<uint>> out_shape_;

    Tensor<T> last_a_;
    Tensor<T> last_z_;

    std::weak_ptr<Layer<T>>   prev_;
    std::shared_ptr<Layer<T>> next_;
};
} // namespace prophecy