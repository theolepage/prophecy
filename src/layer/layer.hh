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

    virtual void backpropagation(Tensor<T>& delta) = 0;

    virtual void compile(std::weak_ptr<Layer<T>> prev, std::shared_ptr<Layer<T>> next)
    {
        this->compiled_ = true;
        this->prev_ = prev;
        this->next_ = next;
    }

    virtual Tensor<T>& cost(const Tensor<T>* const y)
    {
        this->last_a_ -= *y;
        return this->last_a_;
    }

    std::vector<int> get_shape(void) const { return *this->shape_; }
    
    Tensor<T>& get_last_a(void) { return this->last_a_; }

    Tensor<T>& get_last_z(void) { return this->last_z_; }

protected:
    bool compiled_;
    std::shared_ptr<std::vector<int>> shape_;

    Tensor<T> last_a_;
    Tensor<T> last_z_;

    std::weak_ptr<Layer<T>> prev_;
    std::shared_ptr<Layer<T>> next_;
};
