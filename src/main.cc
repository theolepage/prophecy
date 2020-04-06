#include <iostream>

#include "model.hh"
#include "dense_layer.hh"

using namespace std;

using training_set = std::vector<std::shared_ptr<Matrix>>;

static auto get_xor(unsigned a, unsigned b)
{
    auto mx = std::make_shared<Matrix>(2, 1);
    (*mx)(0, 0) = a;
    (*mx)(1, 0) = b;

    auto my = std::make_shared<Matrix>(1, 1);
    (*my)(0, 0) = a^b;

    return std::make_pair(mx, my);
}

static void create_dataset(
        shared_ptr<training_set> x_train,
        shared_ptr<training_set> y_train)
{
    auto a = get_xor(0, 0);
    x_train->emplace_back(a.first);
    y_train->emplace_back(a.second);

    auto b = get_xor(0, 1);
    x_train->emplace_back(b.first);
    y_train->emplace_back(b.second);

    auto c = get_xor(1, 0);
    x_train->emplace_back(c.first);
    y_train->emplace_back(c.second);

    auto d = get_xor(1, 1);
    x_train->emplace_back(d.first);
    y_train->emplace_back(d.second);
}

int main(void)
{
    Model model = Model();

    // Create model
    model.add(make_shared<InputLayer>(2));
    model.add(make_shared<DenseLayer>(2, make_shared<SigmoidActivationFunction>()));
    model.add(make_shared<DenseLayer>(1, make_shared<SigmoidActivationFunction>()));

    // Create dataset
    auto x_train = make_shared<training_set>();
    auto y_train = make_shared<training_set>();
    create_dataset(x_train, y_train);

    // Train model
    model.compile(0.1);
    model.train(x_train, y_train, 10000, 1);

    // Test the model
    for (unsigned i = 0; i < x_train->size(); i++)
    {
        auto x = x_train->at(i);
        auto x_t = x->transpose();
        auto y = model.predict(x);
        cout << "Input:  " << x_t;
        cout << "Output: " << *y << endl;
    }

    return 0;
}
