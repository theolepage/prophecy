#include <iostream>

#include "dense_layer.hh"

DenseLayer::DenseLayer(unsigned nb_neurons,
                       std::shared_ptr<ActivationFunction> activation)
        : HiddenLayer(nb_neurons, activation)
{}

void DenseLayer::compile(unsigned prev_nb_neurons,
                         std::shared_ptr<HiddenLayer> prev,
                         std::shared_ptr<HiddenLayer> next)
{
    // Initialize weights and biases
    weights_ = std::make_shared<Matrix>(get_nb_neurons(), prev_nb_neurons);
    weights_->fill_random();
    biases_ = std::make_shared<Matrix>(get_nb_neurons(), 1);
    biases_->fill_random();

    compiled_ = true;
    prev_ = prev;
    next_ = next;
}

std::shared_ptr<Matrix> DenseLayer::feedforward(std::shared_ptr<Matrix> input, bool training)
{
    auto z = std::make_shared<Matrix>(*weights_ * *input + *biases_);
    auto a = std::make_shared<Matrix>(z->map(activation_->f_));

    if (training)
    {
        last_a_ = a;
        last_z_ = z;
    }

    if (!next_)
        return a;
    return next_->feedforward(a, training);
}

void DenseLayer::backpropagation(std::shared_ptr<Matrix> y)
{
    if (!prev_)
        return;

    if (y)
        delta_ = std::make_shared<Matrix>(Matrix::multiply((*last_a_ - *y), last_z_->map(activation_->fd_)));
    else
        delta_ = std::make_shared<Matrix>(prev_->get_weights()->transpose() * Matrix::multiply(*prev_->get_delta(), last_z_->map(activation_->fd_)));

    delta_biases_ = delta_;
    delta_weights_ = std::make_shared<Matrix>(*delta_ * prev_->get_last_a()->transpose());
    prev_->backpropagation(nullptr);
}

void DenseLayer::update(double learning_rate)
{
    for (unsigned i = 0; i < weights_->get_rows(); i++)
    {
        for (unsigned j = 0; j < weights_->get_cols(); j++)
            (*weights_)(i, j) -= learning_rate * (*delta_weights_)(i, j);
        (*biases_)(i, 0) -= learning_rate * (*delta_biases_)(i, 0);
    }
}
