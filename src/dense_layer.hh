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

    Matrix feedforward(const Matrix& input, bool training);

    void backpropagation();
    void backpropagation(const Matrix& y);

    void compile(std::shared_ptr<Layer> prev,
                 std::shared_ptr<Layer> next);

private:
    bool compiled_;

    std::shared_ptr<Matrix> weights_;
    std::shared_ptr<Matrix> biases_;
    std::shared_ptr<Matrix> delta_weights_;
    std::shared_ptr<Matrix> delta_biases_;

    std::shared_ptr<Layer> prev_;
    std::shared_ptr<Layer> next_;

    ActivationFunction *activation_;
};
