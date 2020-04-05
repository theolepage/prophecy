#include <cmath>

#include "model.hh"

Model::Model()
    : compiled_(false)
{}

Model& Model::add(std::shared_ptr<InputLayer> layer)
{
    input_ = layer;
    return *this;
}

Model& Model::add(std::shared_ptr<HiddenLayer> layer)
{
    layers_.push_back(layer);
    return *this;
}

void Model::compile(double learning_rate)
{
    learning_rate_ = learning_rate;
    compiled_ = true;

    if (layers_.size() >= 2)
    {
        unsigned last = layers_.size() - 1;
        layers_[0]->compile(nullptr, layers_[1]);
        layers_[last]->compile(layers_[last - 1], nullptr);
    }
    for (unsigned i = 1; i < layers_.size() - 1; i++)
        layers_[i]->compile(layers_[i - 1], layers_[i]);
}

void Model::train(std::vector<std::shared_ptr<Matrix>> x,
                  std::vector<std::shared_ptr<Matrix>> y,
                  unsigned epochs,
                  unsigned batch_size)
{
    if (!compiled_)
        throw "Model has not been compiled.";

    for (unsigned epoch = 0; epoch < epochs; epoch++)
    {
        // Determine batches
        unsigned i = 0;
        unsigned nb_batches = ceil(x.size() / batch_size);
        for (unsigned batch = 0; batch < nb_batches; batch++)
        {
            // For each batch, compute delta weights and biases
            for (unsigned k = 0; k < batch_size && i < x.size(); k++)
            {
                layers_[1]->feedforward(x[i], true);
                layers_[layers_.size() - 1]->backpropagation(y[i]);
                i += 1;
            }

            // At the end of batch, update weights_ and biases_
            for (unsigned l = 1; l < layers_.size(); l++)
                layers_[l]->update(learning_rate_);
        }
    }
}

std::shared_ptr<Matrix> Model::predict(std::shared_ptr<Matrix> input)
{
    return layers_[1]->feedforward(input, false);
}
