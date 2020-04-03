#include <cmath>

#include "activation_function.hh"

double SigmoidActivationFunction::function(double x)
{
    return 1 / (1 + exp(-x));
}

double SigmoidActivationFunction::derivative_function(double x)
{
    return function(x) * (1 - function(x));
}
