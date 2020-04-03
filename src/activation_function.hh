#pragma once

#include <functional>

class ActivationFunction
{
public:
    std::function<double(double)> function_;
    std::function<double(double)> derivative_function_;
};

class SigmoidActivationFunction : public ActivationFunction
{
public:
    SigmoidActivationFunction();
};
