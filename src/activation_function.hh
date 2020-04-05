#pragma once

#include <functional>

class ActivationFunction
{
public:
    std::function<double(double)> f_;
    std::function<double(double)> fd_;
};

class SigmoidActivationFunction : public ActivationFunction
{
public:
    SigmoidActivationFunction();
};
