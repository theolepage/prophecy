#pragma once

#include <functional>
#include <cmath>
#include <ctgmath>

namespace prophecy
{
    template <typename T = float>
    class ActivationFunction
    {
    public:
        std::function<T(T)> f_;
        std::function<T(T)> fd_;
    };

    template <typename T = float>
    class LinearActivationFunction final : public ActivationFunction<T>
    {
    public:
        LinearActivationFunction()
        {
            this->f_ = [](const T x) { return x; };
            this->fd_ = [](const T x) { return 1; };
        }
    };

    template <typename T = float>
    class SigmoidActivationFunction final : public ActivationFunction<T>
    {
    public:
        SigmoidActivationFunction()
        {
            this->f_ = [](const T x) { return 1 / (1 + exp(-x)); };
            this->fd_ = [this](const T x) { return this->f_(x) * (1 - this->f_(x)); };
        }
    };

    template <typename T = float>
    class ReLUActivationFunction final : public ActivationFunction<T>
    {
    public:
        ReLUActivationFunction()
        {
            this->f_ = [](const T x) { return (x > 0) ? x : 0; };
            this->fd_ = [this](const T x) { return (x > 0) ? 1 : 0; };
        }
    };
}