#pragma once

#include <ctime>
#include <memory>
#include <string>
#include <vector>

#include "layer/input_layer.hh"
#include "layer/processing_layer.hh"
#include "xtensor/xarray.hpp"

namespace prophecy
{
template <typename T = float>
class Model
{
  public:
    explicit Model();

    virtual ~Model() = default;

    Model& add(const std::shared_ptr<Layer<T>>& layer);

    xt::xarray<T> predict(const xt::xarray<T>& input);

    double get_learning_rate() const;
    void   set_learning_rate(const double lr);

    void train(const xt::xarray<T>& x,
               const xt::xarray<T>& y,
               const uint           batch_size,
               const uint           epochs);

  private:
    bool   compiled_;
    double learning_rate_;

    std::vector<std::shared_ptr<Layer<T>>> layers_;

    void compile();

    const xt::xarray<T> train_batch(const xt::xarray<T>& x,
                                    const xt::xarray<T>& y,
                                    const uint           batch_id,
                                    const uint           batch_size);

    void update_processing_layers() const;
};

template <typename T>
Model<T>::Model()
    : compiled_(false)
    , learning_rate_(0.5)
{
    xt::random::seed(time(NULL));
}

template <typename T>
Model<T>& Model<T>::add(const std::shared_ptr<Layer<T>>& layer)
{
    compiled_ = false;
    layers_.push_back(layer);

    return *this;
}

template <typename T>
xt::xarray<T> Model<T>::predict(const xt::xarray<T>& input)
{
    if (!compiled_)
        compile();

    return layers_[0]->feedforward(input);
}

template <typename T>
double Model<T>::get_learning_rate() const
{
    return learning_rate_;
}

template <typename T>
void Model<T>::set_learning_rate(const double lr)
{
    learning_rate_ = lr;
}

template <typename T>
const xt::xarray<T> Model<T>::train_batch(const xt::xarray<T>& x,
                                          const xt::xarray<T>& y,
                                          const uint           batch_id,
                                          const uint           batch_size)
{
    xt::xarray<T> total_cost = 0;

    uint sample = batch_id * batch_size;

    // For each batch, compute delta weights and biases
    for (uint k = 0; k < batch_size && sample < x.shape()[0]; k++)
    {
        layers_[0]->feedforward(xt::view(x, sample), true);

        auto last_layer = layers_[layers_.size() - 1];
        auto delta      = last_layer->cost(xt::view(y, sample));
        total_cost      = xt::sum(delta);

        last_layer->backpropagation(delta);

        sample++;
    }

    // At the end of batch, update weights_ and biases_
    update_processing_layers();

    return total_cost;
}

template <typename T>
void Model<T>::train(const xt::xarray<T>& x,
                     const xt::xarray<T>& y,
                     const uint           batch_size,
                     const uint           epochs)
{
    if (!compiled_)
        compile();

    const uint batch_count = ceil(1.0f * x.shape()[0] / batch_size);

    for (uint epoch = 0; epoch < epochs; epoch++)
    {
        xt::xarray<T> cost;

        for (uint batch_id = 0; batch_id < batch_count; batch_id++)
            cost = train_batch(x, y, batch_id, batch_size);

        std::cout << "Epoch " << epoch << " completed (loss: " << cost << ")\n";
    }
}

template <typename T>
void Model<T>::compile()
{
    if (layers_.size() < 2)
        throw std::invalid_argument(
            "Model must be composed of at least 2 layers.");

    // Link first layer
    layers_[0]->compile(std::weak_ptr<Layer<T>>(), layers_[1]);

    // Link all layers from last to first
    for (uint i = 1; i < layers_.size() - 1; i++)
        layers_[i]->compile(layers_[i - 1], layers_[i + 1]);

    // Link last layer
    const uint last = layers_.size() - 1;
    layers_[last]->compile(layers_[last - 1], nullptr);

    compiled_ = true;
}

template <typename T>
void Model<T>::update_processing_layers() const
{
    for (uint layer_id = 1; layer_id < layers_.size(); layer_id++)
    {
        auto layer   = layers_[layer_id];
        auto p_layer = std::dynamic_pointer_cast<ProcessingLayer<T>>(layer);
        if (p_layer != nullptr)
            p_layer->update(learning_rate_);
    }
}
} // namespace prophecy