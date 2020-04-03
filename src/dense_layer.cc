#include <iostream>

#include "dense_layer.hh"

DenseLayer::DenseLayer(unsigned nb_neurons, ActivationFunction* activation)
        : Layer(nb_neurons)
        , activation_(activation)
{}

void DenseLayer::compile(std::shared_ptr<Layer> prev,
                         std::shared_ptr<Layer> next)
{
    // Initialize weights and biases
    weights_ = std::make_shared<Matrix>(nb_neurons_, prev->get_nb_neurons());
    weights_->fill_random();
    biases_ = std::make_shared<Matrix>(nb_neurons_, 1);
    biases_->fill_random();

    compiled_ = true;
    prev_ = prev;
    next_ = next;
}

Matrix DenseLayer::feedforward(const Matrix& input, bool training)
{
    Matrix z = *weights_ * input + *biases_;
    Matrix a = z.map(activation_->function_);

    if (training)
    {
        // last_a_ = a;
    }

    if (!next_)
        return a;
    //return next_->feedforward(a, training);
    return Matrix(1, 1);
}

void DenseLayer::backpropagation()
{
    // delta = prev_->weights_.transpose() @ delta * sigmoidprime(z);
    // db = delta;
    // dw = delta @ prev_->a.transpose();
    // prev_->backpropagation();
}

void DenseLayer::backpropagation(const Matrix& y)
{
    (void) y;
    // delta = (a - y) * sigmoidprime(z);
    // db = delta;
    // dw = delta @ prev_->a.transpose();
    // prev_->backpropagation();
}
