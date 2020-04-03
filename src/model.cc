#include <cmath>

#include "model.hh"

Model::Model()
    : compiled_(false)
{}

Model& Model::add(Layer layer)
{
    layers_.push_back(layer);
    return *this;
}

void Model::compile(double learning_rate)
{
    learning_rate_ = learning_rate;
    compiled_ = true;

    for (unsigned i = 1; i < layers_.size(); i++)
    {
        // layers_[i].compile(std::make_shared<Layer>(layers_[i - 1]),
                           //std::make_shared<Layer>(layers_[i]));
    }
}

void Model::train(std::vector<Matrix> x,
           std::vector<Matrix> y,
           unsigned epochs,
           unsigned batch_size)
{
    (void) y;

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
                //layers_[0].feedforward(x[i], true);
                //layers_[layers_.size() - 1].backpropagation(y[i]);
                i += 1;
            }

            // At the end of batch, update weights_ and biases_
            // for (unsigned l = 1; l < layers_.size(); l++)
                // layers_[l].update();
        }
    }
}

Matrix Model::predict(Matrix input)
{
    (void) input;
    //return layers_[1].feedforward(input, false);
    return Matrix(1, 1);
}
