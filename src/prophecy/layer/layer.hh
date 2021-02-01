#pragma once

#include <memory>

#include "xtensor/xarray.hpp"

namespace prophecy
{
template <typename T = float>
class Layer
{
  public:
    explicit Layer(const std::vector<uint>& out_shape);
    explicit Layer();

    virtual ~Layer() = default;

    virtual void compile(std::weak_ptr<Layer<T>>   prev,
                         std::shared_ptr<Layer<T>> next);

    virtual xt::xarray<T> feedforward(const xt::xarray<T>& input,
                                      const bool training = false) = 0;

    virtual void backpropagation(xt::xarray<T>& delta) = 0;

    virtual xt::xarray<T>& cost(const xt::xarray<T>& y);

    std::vector<uint> get_out_shape() const;
    xt::xarray<T>&    get_last_a();
    xt::xarray<T>&    get_last_z();

  protected:
    bool compiled_;

    std::shared_ptr<std::vector<uint>> out_shape_;

    xt::xarray<T> last_a_;
    xt::xarray<T> last_z_;

    std::weak_ptr<Layer<T>>   prev_;
    std::shared_ptr<Layer<T>> next_;
};

template <typename T>
Layer<T>::Layer(const std::vector<uint>& out_shape)
    : compiled_(false)
    , out_shape_(std::make_shared<std::vector<uint>>(out_shape))

{
}

template <typename T>
Layer<T>::Layer()
    : out_shape_(nullptr)
{
}

template <typename T>
void Layer<T>::compile(std::weak_ptr<Layer<T>>   prev,
                       std::shared_ptr<Layer<T>> next)
{
    this->compiled_ = true;
    this->prev_     = prev;
    this->next_     = next;
}

template <typename T>
xt::xarray<T>& Layer<T>::cost(const xt::xarray<T>& y)
{
    this->last_a_ -= y;
    return this->last_a_;
}

template <typename T>
std::vector<uint> Layer<T>::get_out_shape() const
{
    return *this->out_shape_;
}

template <typename T>
xt::xarray<T>& Layer<T>::get_last_a()
{
    return this->last_a_;
}

template <typename T>
xt::xarray<T>& Layer<T>::get_last_z()
{
    return this->last_z_;
}
} // namespace prophecy