#pragma once

#include <memory>
#include <string>
#include <vector>

#include "layer/input_layer.hh"
#include "layer/processing_layer.hh"
#include "tensor/tensor.hh"

namespace prophecy
{
template <typename T = float>
class Model
{
  public:
    Model() : compiled_(false) { srand(time(NULL)); }

    template <typename L>
    Model& add(L layer)
    {
        layers_.emplace_back(std::make_shared<L>(layer));
        return *this;
    }

    Tensor<T> predict(const Tensor<T>& input)
    {
        return layers_[0]->feedforward(input, false);
    }

    virtual ~Model() = default;

    void compile(const double learning_rate)
    {
        if (layers_.size() < 2)
            throw std::invalid_argument(
                "Model must be composed of at least 2 layers to compile.");

        learning_rate_ = learning_rate;
        compiled_      = true;

        // Compile first layer
        layers_[0]->compile(std::weak_ptr<Layer<T>>(), layers_[1]);

        // Compile all layers from last to first
        for (uint i = 1; i < layers_.size() - 1; i++)
            layers_[i]->compile(layers_[i - 1], layers_[i + 1]);

        // Compile last layer
        const uint last = layers_.size() - 1;
        layers_[last]->compile(layers_[last - 1], nullptr);
    }

    void train(std::vector<Tensor<T>>& x,
               std::vector<Tensor<T>>& y,
               const uint              epochs,
               const uint              batch_size)
    {
        if (!compiled_)
            throw "Model has not been compiled.";

        for (uint epoch = 0; epoch < epochs; epoch++)
        {
            T total_cost;

            // Determine batches
            uint       i          = 0;
            const uint nb_batches = ceil(1.0f * x.size() / batch_size);
            for (uint batch = 0; batch < nb_batches; batch++)
            {
                // For each batch, compute delta weights and biases
                for (uint k = 0; k < batch_size && i < x.size(); k++)
                {
                    layers_[0]->feedforward(x[i], true);

                    auto last_layer = layers_[layers_.size() - 1];
                    auto delta      = last_layer->cost(y[i]);
                    total_cost      = delta.sum()({0});
                    last_layer->backpropagation(delta);

                    i++;
                }

                // At the end of batch, update weights_ and biases_
                for (uint l = 1; l < layers_.size(); l++)
                {
                    auto layer = std::dynamic_pointer_cast<ProcessingLayer<T>>(
                        layers_[l]);
                    if (layer != nullptr)
                        layer->update(learning_rate_);
                }
            }

            std::cout << "Epoch " << epoch << " completed (loss: " << total_cost
                      << ")\n";
        }
    }

    void summary();
    void save(const std::string& path);
    void load(const std::string& path);

  private:
    bool                                   compiled_;
    double                                 learning_rate_;
    std::vector<std::shared_ptr<Layer<T>>> layers_;
};
} // namespace prophecy