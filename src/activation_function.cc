#include <cmath>

#include "activation_function.hh"

SigmoidActivationFunction::SigmoidActivationFunction()
{
    function_ = [](double x) {
        return 1 / (1 + exp(-x));
    };

    derivative_function_ = [this](double x) {
        return function_(x) * (1 - function_(x));
    };
}
