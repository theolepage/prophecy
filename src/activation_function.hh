#pragma once

class ActivationFunction
{
public:
    virtual double function(double x) = 0;
    virtual double derivative_function(double x) = 0;
};

class SigmoidActivationFunction : public ActivationFunction
{
public:
    double function(double x);
    double derivative_function(double x);
};
