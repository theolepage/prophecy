#include <iostream>
#include <memory>

#include "pybind11/pybind11.h"
#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"

#include "layer/dense_layer.hh"
#include "model/model.hh"
#include "python_bindings.hh"

#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"

namespace py = pybind11;
using namespace prophecy;
using model_type = float;

static void xor_example(void)
{
    Model model = Model();

    auto s = std::make_shared<SigmoidActivationFunction<model_type>>();

    // Create model
    std::vector<uint> input_shape = {2};
    model.add(std::make_shared<InputLayer<model_type>>(input_shape));
    model.add(std::make_shared<DenseLayer<model_type>>(2, s));
    model.add(std::make_shared<DenseLayer<model_type>>(1, s));

    // Create dataset
    xt::xarray<model_type> x = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f};
    xt::xarray<model_type> y = {0 ^ 0, 0 ^ 1, 1 ^ 0, 1 ^ 1};
    x.reshape({4, 2, 1});
    y.reshape({4, 1, 1});

    // Train model
    model.set_learning_rate(0.1);
    model.train(x, y, 1, 10000);

    // Test the model
    for (size_t i = 0; i < x.shape().at(0); i++)
    {
        auto x_i = xt::view(x, i);
        auto y   = model.predict(x_i);
        std::cout << "Input:  " << xt::transpose(x_i) << std::endl;
        std::cout << "Output: " << y << std::endl;
    }
}

PYBIND11_MODULE(prophecy, m)
{
    xt::import_numpy();

    m.doc() = "Python bindings for Prophecy";

    py::class_<py_model>(m, "Model")
        .def(py::init<>())
        .def("predict", &py_model::predict)
        .def("train", &py_model::train)
        .def("add", &py_model::add)
        .def("set_learning_rate", &py_model::set_learning_rate)
        .def("get_learning_rate", &py_model::get_learning_rate);

    py::class_<py_layers>(m, "ProphecyLayer");
    py::class_<py_activations>(m, "ProphecyActivation");

    m.def_submodule("layers")
        .def("input", &py_layers::input)
        .def("dense", &py_layers::dense);

    m.def_submodule("activations")
        .def("linear", &py_activations::linear)
        .def("sigmoid", &py_activations::sigmoid);
}