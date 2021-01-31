#include <iostream>
#include <memory>

#include "kernel.cuh"
#include "layer/dense_layer.hh"
#include "model/model.hh"
#include "tensor/tensor.hh"

#include "xtensor-blas/xlinalg.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"

using namespace prophecy;
using model_type = float;

auto get_xor(const uint a, const uint b)
{
    auto mx    = Tensor<model_type>({2u, 1u});
    mx(0u, 0u) = a;
    mx(1u, 0u) = b;

    auto my    = Tensor<model_type>({1u, 1u});
    my(0u, 0u) = a ^ b;

    return std::make_pair(mx, my);
}

static void load_xor(std::vector<Tensor<model_type>>& x,
                     std::vector<Tensor<model_type>>& y)
{
    auto a = get_xor(0, 0);
    x.emplace_back(a.first);
    y.emplace_back(a.second);

    auto b = get_xor(0, 1);
    x.emplace_back(b.first);
    y.emplace_back(b.second);

    auto c = get_xor(1, 0);
    x.emplace_back(c.first);
    y.emplace_back(c.second);

    auto d = get_xor(1, 1);
    x.emplace_back(d.first);
    y.emplace_back(d.second);
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
    std::vector<Tensor<model_type>> x;
    std::vector<Tensor<model_type>> y;
    load_xor(x, y);

    // Train model
    model.set_learning_rate(0.1);
    model.train(x, y, 1, 10000);

    // Test the model
    for (size_t i = 0; i < x.size(); i++)
    {
        auto x_i = x.at(i);
        auto x_t = x_i.transpose();
        auto y   = model.predict(x_i);
        std::cout << "Input:  " << x_t;
        std::cout << "Output: " << y << std::endl;
    }
}

int main(void)
{
    xor_example();

    xt::xarray<double> arr1{{1.0, 2.0, 3.0}, {2.0, 5.0, 7.0}, {2.0, 5.0, 7.0}};
    std::cout << arr1;

    xt::xarray<float> a  = xt::ones<float>({5, 3});
    auto              r1 = xt::linalg::dot(a, xt::transpose(a));

    return 0;
}