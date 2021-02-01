#include <iostream>
#include <memory>

#include "layer/dense_layer.hh"
#include "model/model.hh"

#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"

#include "pybind11/pybind11.h"
#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"

using namespace prophecy;
using model_type = float;

auto store_xor(const uint                           a,
               const uint                           b,
               std::vector<xt::xarray<model_type>>& x,
               std::vector<xt::xarray<model_type>>& y)
{
    float a_f       = a;
    float b_f       = b;
    float a_xor_b_f = a ^ b;

    xt::xarray<model_type> mx = {a_f, b_f};
    mx.reshape({2, 1});
    xt::xarray<model_type> my = {a_xor_b_f};
    my.reshape({1, 1});

    x.push_back(mx);
    y.push_back(my);
}

static void xor_example(void)
{
    Model model = Model();

    SigmoidActivationFunction s = SigmoidActivationFunction();

    // Create model
    model.add_layer(InputLayer({2}));
    model.add_layer(DenseLayer(2, s));
    model.add_layer(DenseLayer(1, s));

    // Create dataset
    std::vector<xt::xarray<model_type>> x;
    std::vector<xt::xarray<model_type>> y;
    store_xor(0, 0, x, y);
    store_xor(0, 1, x, y);
    store_xor(1, 0, x, y);
    store_xor(1, 1, x, y);

    // Train model
    model.set_learning_rate(0.1);
    model.train(x, y, 1, 10000);

    // Test the model
    for (size_t i = 0; i < x.size(); i++)
    {
        auto x_i = x.at(i);
        auto x_t = xt::transpose(x_i);
        auto y   = model.predict(x_i);
        std::cout << "Input:  " << x_t;
        std::cout << "Output: " << y << std::endl;
    }
}

PYBIND11_MODULE(prophecy, m)
{
    xt::import_numpy();

    m.doc() = "Prophecy neural networks framework";

    m.def("xor", xor_example, "xor");
}