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

    xt::xarray<T> evaluate(const xt::xarray<T>& x, const xt::xarray<T>& y);

    double get_learning_rate() const;
    void   set_learning_rate(const double lr);

    void summary();

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
                                    const uint           batch_size);

    void print_progress(int           epoch,
                        int           epochs,
                        int           batch,
                        int           batches,
                        xt::xarray<T> loss);

    void update_processing_layers(const uint batch_size) const;
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
xt::xarray<T> Model<T>::evaluate(const xt::xarray<T>& x, const xt::xarray<T>& y)
{
    if (!compiled_)
        compile();

    xt::xarray<T> loss = 0;

    for (uint i = 0; i < x.shape()[0]; i++)
    {
        layers_[0]->feedforward(xt::view(x, i));

        auto last_layer = layers_[layers_.size() - 1];
        auto delta      = last_layer->cost(xt::view(y, i));
        loss += xt::sum(delta);
    }

    loss /= x.shape()[0];
    return loss;
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
void Model<T>::summary()
{
    const uint name_length   = 12;
    const uint shape_length  = 18;
    const uint params_length = 10;
    const uint total_length  = name_length + shape_length + params_length;

    if (!compiled_)
        compile();

    uint total_params_count = 0;

    // Print header
    std::cout << "Model:" << std::endl;
    std::cout << std::string(total_length, '_') << std::endl;
    std::cout << "Layer" << std::string(name_length - 5, ' ');
    std::cout << "Output shape" << std::string(shape_length - 12, ' ');
    std::cout << "Param #" << std::string(params_length - 7, ' ');
    std::cout << std::endl << std::string(total_length, '=') << std::endl;

    for (uint i = 0; i < layers_.size(); i++)
    {
        // Print name
        const std::string name = layers_[i]->get_name().substr(0, name_length);
        std::cout << name << std::string(name_length - name.length() + 1, ' ');

        // Print output shape
        const std::vector<uint> shape = layers_[i]->get_out_shape();
        std::string             shape_str;
        shape_str.append("(");
        for (uint j = 0; j < shape.size(); j++)
        {
            shape_str.append(std::to_string(shape.at(j)));
            shape_str.append(((j == shape.size() - 1) ? "" : ", "));
        }
        shape_str.append(")");
        shape_str = shape_str.substr(0, shape_length);
        std::cout << shape_str
                  << std::string(shape_length - shape_str.length() + 1, ' ');

        // Print params count
        const uint        params_count     = layers_[i]->get_params_count();
        const std::string params_count_str = std::to_string(params_count);
        std::cout << params_count_str.substr(0, params_length);
        total_params_count += params_count;

        std::cout << std::endl;
    }

    // Print footer
    std::cout << std::string(total_length, '=') << std::endl;
    std::cout << "Total params: " << total_params_count << std::endl;
    std::cout << std::string(total_length, '_') << std::endl;
}

template <typename T>
const xt::xarray<T> Model<T>::train_batch(const xt::xarray<T>& x,
                                          const xt::xarray<T>& y,
                                          const uint           batch_size)
{
    // Backpropagation over layers
    layers_[0]->feedforward(x, true);
    auto delta = layers_[layers_.size() - 1]->cost(y);
    layers_[layers_.size() - 1]->backpropagation(delta);

    // At the end update weights and biases
    update_processing_layers(batch_size);

    // FIXME compute correctly loss
    xt::xarray<T> total_loss = xt::sum(delta) / batch_size;
    return total_loss;
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

    std::cout << "Training:" << std::endl;

    for (uint epoch = 0; epoch < epochs; epoch++)
    {
        for (uint batch = 0; batch < batch_count; batch++)
        {
            auto rng = xt::range(batch * batch_size, (batch + 1) * batch_size);
            xt::xarray<T> x_batch = xt::view(x, rng, xt::all());
            xt::xarray<T> y_batch = xt::view(y, rng, xt::all());

            xt::xarray<T> loss = train_batch(x_batch, y_batch, batch_size);
            print_progress(epoch, epochs, batch, batch_count, loss);
        }
    }

    std::cout << std::endl;
}

template <typename T>
void Model<T>::print_progress(int           epoch,
                              int           epochs,
                              int           batch,
                              int           batches,
                              xt::xarray<T> loss)
{
    epoch++;
    batch++;

    // Print epoch number and progress on training set
    std::cout << "Epoch " << epoch << "/" << epochs << " - ";
    std::cout << batch << "/" << batches << " [";

    // Print progress bar
    int progress = 30.0f * batch / batches;
    for (int p = 0; p < progress; p++)
        std::cout << (p == progress - 1 ? ">" : "=");
    for (int p = 0; p < 30 - progress; p++)
        std::cout << ".";
    std::cout << "]";

    // Print loss on training set
    std::cout << " - loss: " << loss(0) << "\r" << std::flush;
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
void Model<T>::update_processing_layers(const uint batch_size) const
{
    for (uint layer_id = 1; layer_id < layers_.size(); layer_id++)
    {
        auto layer   = layers_[layer_id];
        auto p_layer = std::dynamic_pointer_cast<ProcessingLayer<T>>(layer);
        if (p_layer != nullptr)
            p_layer->update(learning_rate_, batch_size);
    }
}
} // namespace prophecy