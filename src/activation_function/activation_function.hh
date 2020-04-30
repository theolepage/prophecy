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
class LinearActivationFunction : public ActivationFunction<T>
{
public:
    LinearActivationFunction()
    {
        this->f_ = [](T x) { return x; };
        this->fd_ = [this](T x) { return 1; };
    }
};

template <typename T>
class SigmoidActivationFunction : public ActivationFunction<T>
{
public:
    SigmoidActivationFunction()
    {
        this->f_ = [](T x) { return 1 / (1 + exp(-x)); };
        this->fd_ = [this](T x) { return this->f_(x) * (1 - this->f_(x)); };
    }
};

template <typename T>
class ReLUActivationFunction : public ActivationFunction<T>
{
public:
    ReLUActivationFunction()
    {
        this->f_ = [](T x) { return (x > 0) ? x : 0; };
        this->fd_ = [this](T x) { return (x > 0) ? 1 : 0; };
    }
};
