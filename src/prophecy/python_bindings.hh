#pragma once

#include "pybind11/iostream.h"
#include "pybind11/pybind11.h"

#include "layer/dense_layer.hh"
#include "model/model.hh"

namespace prophecy
{
using model_type = float;

struct py_activations
{
    py_activations(const std::shared_ptr<ActivationFunction<model_type>> a)
        : ptr_(a)
    {
    }

    const std::shared_ptr<ActivationFunction<model_type>> ptr_;

    static py_activations linear()
    {
        return py_activations(
            std::make_shared<LinearActivationFunction<model_type>>());
    }

    static py_activations sigmoid()
    {
        return py_activations(
            std::make_shared<SigmoidActivationFunction<model_type>>());
    }
};

struct py_layers
{
    py_layers(std::shared_ptr<Layer<model_type>> layer)
        : ptr_(layer)
    {
    }

    std::shared_ptr<Layer<model_type>> ptr_;

    static py_layers input(uint nb_neurons)
    {
        std::vector<uint> shape = {nb_neurons};

        auto l = std::make_shared<InputLayer<model_type>>(shape);
        return py_layers(l);
    }

    static py_layers dense(uint nb_neurons, py_activations activation)
    {
        auto l = std::make_shared<DenseLayer<model_type>>(nb_neurons,
                                                          activation.ptr_);
        return py_layers(l);
    }
};

struct py_model
{
    py_model() {}

    Model<model_type> model_;

    void add(const py_layers& layer) { model_.add(layer.ptr_); }

    xt::pyarray<model_type> predict(const xt::pyarray<model_type>& input)
    {
        return model_.predict(input);
    }

    xt::pyarray<model_type> evaluate(const xt::pyarray<model_type>& x,
                                     const xt::pyarray<model_type>& y)
    {
        return model_.evaluate(x, y);
    }

    void train(const xt::pyarray<model_type>& x,
               const xt::pyarray<model_type>& y,
               const uint                     batch_size,
               const uint                     epochs)
    {
        pybind11::scoped_ostream_redirect stream();
        model_.train(x, y, batch_size, epochs);
    }

    void summary()
    {
        pybind11::scoped_ostream_redirect stream();
        model_.summary();
    }

    double get_learning_rate() const { return model_.get_learning_rate(); }
    void   set_learning_rate(const double lr) { model_.set_learning_rate(lr); }
};
} // namespace prophecy