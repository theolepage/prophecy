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
    std::shared_ptr<Matrix> get_last_a();
    std::shared_ptr<Matrix> get_delta();

    virtual std::shared_ptr<Matrix> feedforward(std::shared_ptr<Matrix> input, bool training) = 0;

    virtual void backpropagation(std::shared_ptr<Matrix> y) = 0;

    virtual void update(double learning_rate) = 0;

    virtual void compile(unsigned prev_nb_neurons,
                         std::shared_ptr<HiddenLayer> prev,
                         std::shared_ptr<HiddenLayer> next) = 0;

protected:
    bool compiled_;

    std::shared_ptr<Matrix> weights_;
    std::shared_ptr<Matrix> biases_;

    std::shared_ptr<Matrix> delta_weights_;
    std::shared_ptr<Matrix> delta_biases_;
    std::shared_ptr<Matrix> last_a_;
    std::shared_ptr<Matrix> last_z_;
    std::shared_ptr<Matrix> delta_;

    std::shared_ptr<HiddenLayer> prev_;
    std::shared_ptr<HiddenLayer> next_;

    std::shared_ptr<ActivationFunction> activation_;
};
