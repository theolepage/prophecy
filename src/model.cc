#include "model.hh"

Model::Model()
    : compiled_(false)
{}

Model& Model::add(Layer layer)
{
    layers_.push_back(layer);
}

void Model::compile(double learning_rate)
{
    learning_rate_ = learning_rate;
    compiled_ = true;

    for (unsigned i = 1; i < layers_.size(); i++)
        layers_[i].compile(layers_[i - 1]);
}

void train(std::vector<Matrix> x,
            std::vector<Matrix y,
            unsigned epoch,
            unsigned batch_size)
{

}

Matrix predict(Matrix x)
{
    for (Layer layer : layers_)
        x = layer.compute_activations(x);
    return x;
}
