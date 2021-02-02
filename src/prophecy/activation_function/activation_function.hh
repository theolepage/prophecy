#pragma once

#include <cmath>
#include <ctgmath>
#include <functional>

namespace prophecy
{
template <typename T = float>
class ActivationFunction
{
  public:
    virtual const std::function<T(T)> get_f() const  = 0;
    virtual const std::function<T(T)> get_fd() const = 0;
};

template <typename T = float>
class LinearActivationFunction final : public ActivationFunction<T>
{
  public:
    const std::function<T(T)> get_f() const;
    const std::function<T(T)> get_fd() const;
};

template <typename T = float>
class SigmoidActivationFunction final : public ActivationFunction<T>
{
  public:
    const std::function<T(T)> get_f() const;
    const std::function<T(T)> get_fd() const;
};

template <typename T = float>
class ReLUActivationFunction final : public ActivationFunction<T>
{
  public:
    const std::function<T(T)> get_f() const;
    const std::function<T(T)> get_fd() const;
};

template <typename T>
const std::function<T(T)> LinearActivationFunction<T>::get_f() const
{
    return [](const T x) { return x; };
}

template <typename T>
const std::function<T(T)> LinearActivationFunction<T>::get_fd() const
{
    return [](const T) { return 1; };
}

template <typename T>
const std::function<T(T)> SigmoidActivationFunction<T>::get_f() const
{
    return [](const T x) { return 1 / (1 + exp(-x)); };
}

template <typename T>
const std::function<T(T)> SigmoidActivationFunction<T>::get_fd() const
{
    return [this](const T x) { return get_f()(x) * (1 - get_f()(x)); };
}

template <typename T>
const std::function<T(T)> ReLUActivationFunction<T>::get_f() const
{
    return [](const T x) { return (x > 0) ? x : 0; };
}

template <typename T>
const std::function<T(T)> ReLUActivationFunction<T>::get_fd() const
{
    return [](const T x) { return (x > 0) ? 1 : 0; };
}
} // namespace prophecy