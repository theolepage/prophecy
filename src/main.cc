#include <iostream>
#include <memory>

#include "model/model.hh"
#include "layer_implem/dense_layer.hh"

using model_type = float;
using training_set = std::vector<Matrix<model_type>>;

static auto get_xor(unsigned a, unsigned b)
{
    auto mx = Matrix<model_type>(2, 1);
    mx(0, 0) = a;
    mx(1, 0) = b;

    auto my = Matrix<model_type>(1, 1);
    my(0, 0) = a^b;

    return std::make_pair(mx, my);
}

static void create_dataset(
        training_set& x_train,
        training_set& y_train)
{
    auto a = get_xor(0, 0);
    x_train.emplace_back(a.first);
    y_train.emplace_back(a.second);

    auto b = get_xor(0, 1);
    x_train.emplace_back(b.first);
    y_train.emplace_back(b.second);

    auto c = get_xor(1, 0);
    x_train.emplace_back(c.first);
    y_train.emplace_back(c.second);

    auto d = get_xor(1, 1);
    x_train.emplace_back(d.first);
    y_train.emplace_back(d.second);
}

int main(void)
{
    Model<model_type> model = Model<model_type>();
    SigmoidActivationFunction s = SigmoidActivationFunction<model_type>();

    // Create model
    model.add(new InputLayer<model_type>(2));
    model.add(new DenseLayer<model_type>(2, s));
    model.add(new DenseLayer<model_type>(1, s));

    // Create dataset
    auto x_train = training_set();
    auto y_train = training_set();
    create_dataset(x_train, y_train);

    // Train model
    model.compile(0.1);
    model.train(x_train, y_train, 10000, 1);

    // Test the model
    for (size_t i = 0; i < x_train.size(); i++)
    {
        auto x = x_train.at(i);
        auto x_t = x.transpose();
        auto y = model.predict(x);
        std::cout << "Input:  " << x_t;
        std::cout << "Output: " << y << std::endl;
    }

    return 0;
}
