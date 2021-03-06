#pragma once

#include <memory>
#include <string>
#include <vector>

#include "../layer/input_layer.hh"
#include "../layer/hidden_layer.hh"

template <typename T = float>
class Model
{
public:
    Model() : compiled_(false)
    {
        srand(time(NULL));
    }

    Model& add(Layer<T>* layer)
    {
        layers_.emplace_back(layer);
        return *this;
    }

    Matrix<T> predict(const Matrix<T>& input)
    {
        return layers_[0]->feedforward(input, false);
    }

    virtual ~Model() = default;

    void compile(T learning_rate)
    {
        learning_rate_ = learning_rate;
        compiled_ = true;

        if (layers_.size() >= 2)
        {
            int last = layers_.size() - 1;
            layers_[0]->compile(std::weak_ptr<Layer<T>>(), layers_[1]);
            layers_[last]->compile(layers_[last - 1], nullptr);
        }
        for (size_t i = 1; i < layers_.size() - 1; i++)
            layers_[i]->compile(layers_[i - 1], layers_[i + 1]);
    }

    void train(std::vector<Matrix<T>>& x,
                std::vector<Matrix<T>>& y,
                int epochs,
                int batch_size)
    {
        if (!compiled_)
            throw "Model has not been compiled.";

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            // Determine batches
            int i = 0;
            int nb_batches = ceil(x.size() / batch_size);
            for (int batch = 0; batch < nb_batches; batch++)
            {
                // For each batch, compute delta weights and biases
                for (int k = 0; k < batch_size && i < static_cast<int>(x.size()); k++)
                {
                    layers_[0]->feedforward(x[i], true);
                    layers_[layers_.size() - 1]->backpropagation(&y[i]);
                    ++i;
                }

                // At the end of batch, update weights_ and biases_
                for (size_t l = 1; l < layers_.size(); l++)
                {
                    auto layer = std::dynamic_pointer_cast<HiddenLayer<T>>(layers_[l]);
                    layer->update(learning_rate_);
                }
            }
        }
    }

    void summary();
    void save(const std::string& path);
    void load(const std::string& path);

private:
    bool compiled_;
    T learning_rate_;
    std::vector<std::shared_ptr<Layer<T>>> layers_;
};
