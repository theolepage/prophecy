#pragma once

#include <functional>
#include <cmath>
#include <ctgmath>

template <typename T>
class ActivationFunction
{
public:
    std::function<T(T)> f_;
    std::function<T(T)> fd_;
};

template <typename T>
class SigmoidActivationFunction : public ActivationFunction<T>
{
public:
    SigmoidActivationFunction()
    {
        this->f_ = [](T x) {
            return 1 / (1 + exp(-x));
        };

        this->fd_ = [this](T x) {
            return this->f_(x) * (1 - this->f_(x));
        };
    }
};
