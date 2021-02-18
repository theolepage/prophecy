#include <iostream>
#include <memory>

#include "pybind11/pybind11.h"
#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"

#include "layer/dense_layer.hh"
#include "model/model.hh"
#include "python_bindings.hh"

using namespace prophecy;

PYBIND11_MODULE(prophecy, m)
{
    xt::import_numpy();

    m.doc() = "Python bindings for Prophecy";

    pybind11::class_<py_model>(m, "Model")
        .def(pybind11::init<>())
        .def("predict", &py_model::predict)
        .def("evaluate", &py_model::evaluate)
        .def("train", &py_model::train)
        .def("add", &py_model::add)
        .def("summary", &py_model::summary)
        .def("set_learning_rate", &py_model::set_learning_rate)
        .def("get_learning_rate", &py_model::get_learning_rate);

    pybind11::class_<py_layers>(m, "ProphecyLayer");
    m.def_submodule("layers")
        .def("input", &py_layers::input)
        .def("dense",
             &py_layers::dense,
             pybind11::arg("nb_neurons"),
             pybind11::arg("activation"),
             pybind11::arg("init") = "xavier");

    pybind11::class_<py_activations>(m, "ProphecyActivation");
    m.def_submodule("activations")
        .def("linear", &py_activations::linear)
        .def("sigmoid", &py_activations::sigmoid);
}
