#include "gtest/gtest.h"

#include "layer/dense_layer.hh"
#include "model/model.hh"
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"

using namespace prophecy;

TEST(test_xor, check_results)
{
    Model model = Model();

    auto s = std::make_shared<SigmoidActivationFunction<float>>();

    // Create model
    std::vector<uint> input_shape = {2};
    model.add(std::make_shared<InputLayer<float>>(input_shape));
    model.add(std::make_shared<DenseLayer<float>>(2, s));
    model.add(std::make_shared<DenseLayer<float>>(1, s));

    // Create dataset
    xt::xarray<float> x = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f};
    xt::xarray<float> y = {0 ^ 0, 0 ^ 1, 1 ^ 0, 1 ^ 1};
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

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}