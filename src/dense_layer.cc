#include <iostream>

#include "dense_layer.hh"

DenseLayer::DenseLayer(unsigned nb_neurons, ActivationFunction* activation)
        : Layer(nb_neurons)
        , activation_(activation)
{}

void DenseLayer::compile(Layer prev)
{
    weights_ = std::make_shared<Matrix>(nb_neurons_, prev.get_nb_neurons());
    weights_->fill_random();
    biases_ = std::make_shared<Matrix>(nb_neurons_, 1);
    biases_->fill_random();

    compiled_ = true;
}

Matrix DenseLayer::compute_activations(const Matrix& input)
{
    return (*weights_ * input + *biases_).map(activation_->function_);
}
