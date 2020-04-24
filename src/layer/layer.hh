#pragma once

#include <memory>

#include "../tensor/tensor.hh"

template <typename T>
class Layer
{
public:
    Layer(const std::vector<int>& shape)
        : shape_(std::make_shared<std::vector<int>>(shape))
    {}

    virtual ~Layer() = default;

    virtual Tensor<T> feedforward(const Tensor<T>& input, bool training) = 0;

    virtual void backpropagation(const Tensor<T>* const y) = 0;

    virtual void compile(std::weak_ptr<Layer<T>> prev,
                         std::shared_ptr<Layer<T>> next)
    {
        compiled_ = true;
        prev_ = prev;
        next_ = next;
    }

    std::vector<int> get_shape(void) const { return *shape_; }
    
    Tensor<T>& get_last_a(void) { return last_a_; }

    Tensor<T>& get_last_z(void) { return last_z_; }

protected:
    bool compiled_;
    std::shared_ptr<std::vector<int>> shape_;

    Tensor<T> last_a_;
    Tensor<T> last_z_;

    std::weak_ptr<Layer<T>> prev_;
    std::shared_ptr<Layer<T>> next_;
};
