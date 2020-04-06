#pragma once

#include <memory>
#include <functional>

#include "layer.hh"
#include "matrix.hh"
#include "activation_function.hh"

class HiddenLayer : public Layer
{
public:
    HiddenLayer(unsigned nb_neurons, std::shared_ptr<ActivationFunction> activation);

    std::shared_ptr<Matrix> get_weights();
    std::shared_ptr<Matrix> get_biases();

    virtual void update(double learning_rate) = 0;

protected:
    std::shared_ptr<Matrix> weights_;
    std::shared_ptr<Matrix> biases_;

    std::shared_ptr<Matrix> delta_weights_;
    std::shared_ptr<Matrix> delta_biases_;

    std::shared_ptr<ActivationFunction> activation_;
};
