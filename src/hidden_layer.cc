#include "hidden_layer.hh"

HiddenLayer::HiddenLayer(unsigned nb_neurons, std::shared_ptr<ActivationFunction> activation)
    : Layer(nb_neurons)
    , activation_(activation)
{}

std::shared_ptr<Matrix> HiddenLayer::get_weights()
{
    return weights_;
}

std::shared_ptr<Matrix> HiddenLayer::get_biases()
{
    return biases_;
}
