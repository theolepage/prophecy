#pragma once

#include "hidden_layer.hh"

class DenseLayer : public HiddenLayer
{
public:
    DenseLayer(unsigned nb_neurons,
               std::shared_ptr<ActivationFunction> activation);

    std::shared_ptr<Matrix> feedforward(std::shared_ptr<Matrix> input, bool training);

    void backpropagation(std::shared_ptr<Matrix> y);

    void update(double learning_rate);

    void compile(unsigned prev_nb_neurons,
                 std::shared_ptr<HiddenLayer> prev,
                 std::shared_ptr<HiddenLayer> next);
};
