#pragma once

#include <memory>
#include <functional>

#include "layer.hh"
#include "matrix.hh"
#include "activation_function.hh"

class DenseLayer : public Layer
{
public:
    DenseLayer(unsigned nb_neurons, ActivationFunction *activation);

    void compile(Layer prev);

    Matrix compute_activations(const Matrix& input);

private:
    bool compiled_;
    std::shared_ptr<Matrix> weights_;
    std::shared_ptr<Matrix> biases_;
    ActivationFunction *activation_;
};
