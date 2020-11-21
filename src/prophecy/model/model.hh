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
    Model() : compiled_(false), learning_rate_(0.5) { srand(time(NULL)); }

    virtual ~Model() = default;

    template <typename L>
    Model& add_layer(L layer)
    {
        compiled_ = false;
        layers_.emplace_back(std::make_shared<L>(layer));

        return *this;
    }

    Tensor<T> predict(const Tensor<T>& input)
    {
        if (!compiled_)
            compile();

        return layers_[0]->feedforward(input, false);
    }

    double get_learning_rate() const { return learning_rate_; }

    void set_learning_rate(const double lr) { learning_rate_ = lr; }

    const T train_batch(std::vector<Tensor<T>>& x,
                        std::vector<Tensor<T>>& y,
                        const uint              batch_id,
                        const uint              batch_size)
    {
        T    total_cost;
        uint sample = batch_id * batch_size;

        // For each batch, compute delta weights and biases
        for (uint k = 0; k < batch_size && sample < x.size(); k++)
        {
            layers_[0]->feedforward(x[sample], true);

            auto last_layer = layers_[layers_.size() - 1];
            auto delta      = last_layer->cost(y[sample]);
            total_cost      = delta.sum()({0});

            last_layer->backpropagation(delta);

            sample++;
        }

        // At the end of batch, update weights_ and biases_
        update_processing_layers();

        return total_cost;
    }

    void train(std::vector<Tensor<T>>& x,
               std::vector<Tensor<T>>& y,
               const uint              batch_size,
               const uint              epochs)
    {
        if (!compiled_)
            compile();

        const uint batch_count = ceil(1.0f * x.size() / batch_size);

        for (uint epoch = 0; epoch < epochs; epoch++)
        {
            T cost;

            for (uint batch_id = 0; batch_id < batch_count; batch_id++)
                cost = train_batch(x, y, batch_id, batch_size);

            std::cout << "Epoch " << epoch << " completed (loss: " << cost
                      << ")\n";
        }
    }

  private:
    bool   compiled_;
    double learning_rate_;

    std::vector<std::shared_ptr<Layer<T>>> layers_;

    void compile()
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

    void update_processing_layers() const
    {
        for (uint layer_id = 1; layer_id < layers_.size(); layer_id++)
        {
            auto layer   = layers_[layer_id];
            auto p_layer = std::dynamic_pointer_cast<ProcessingLayer<T>>(layer);
            if (p_layer != nullptr)
                p_layer->update(learning_rate_);
        }
    }
};
} // namespace prophecy