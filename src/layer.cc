#include "layer.hh"

Layer::Layer(unsigned nb_neurons)
    : nb_neurons_(nb_neurons)
{}

unsigned Layer::get_nb_neurons()
{
    return nb_neurons_;
}

std::shared_ptr<Matrix> Layer::get_last_a()
{
    return last_a_;
}

std::shared_ptr<Matrix> Layer::get_last_z()
{
    return last_z_;
}

std::shared_ptr<Matrix> Layer::get_delta()
{
    return delta_;
}

std::shared_ptr<Matrix> Layer::feedforward(std::shared_ptr<Matrix> input, bool training)
{
    last_a_ = input;
    return next_->feedforward(input, training);
}

void Layer::backpropagation(std::shared_ptr<Matrix> y)
{
    if (!prev_)
        return;

    (void) y;
    return prev_->backpropagation(nullptr);
}

void Layer::compile(std::shared_ptr<Layer> prev,
                    std::shared_ptr<Layer> next)
{
    compiled_ = true;
    prev_ = prev;
    next_ = next;
}
