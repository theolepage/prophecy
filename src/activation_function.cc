#include <cmath>

#include "activation_function.hh"

SigmoidActivationFunction::SigmoidActivationFunction()
{
    f_ = [](double x) {
        return 1 / (1 + exp(-x));
    };

    fd_ = [this](double x) {
        return f_(x) * (1 - f_(x));
    };
}
