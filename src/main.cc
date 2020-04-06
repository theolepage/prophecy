#include <iostream>

#include "model.hh"
#include "dense_layer.hh"

using namespace std;

int main(void)
{
    srand(time(NULL));

    Model model = Model();

    // Create model
    model.add(make_shared<InputLayer>(2));
    model.add(make_shared<DenseLayer>(4, make_shared<SigmoidActivationFunction>()));
    model.add(make_shared<DenseLayer>(2, make_shared<SigmoidActivationFunction>()));

    // Create dataset
    auto x_train = std::make_shared<std::vector<std::shared_ptr<Matrix>>>();
    auto y_train = std::make_shared<std::vector<std::shared_ptr<Matrix>>>();
    for (unsigned i = 0; i < 10000; i++)
    {
        unsigned rand = std::rand() % 4;
        auto mx = std::make_shared<Matrix>(2, 1);
        auto my = std::make_shared<Matrix>(2, 1);

        if (rand == 0)
        {
            (*mx)(0, 0) = 1;
            (*mx)(1, 0) = 0;

            (*my)(0, 0) = 1;
            (*my)(1, 0) = 0;
        }
        else if (rand == 1)
        {
            (*mx)(0, 0) = 0;
            (*mx)(1, 0) = 1;

            (*my)(0, 0) = 1;
            (*my)(1, 0) = 0;
        }
        else if (rand == 2)
        {
            (*mx)(0, 0) = 0;
            (*mx)(1, 0) = 0;

            (*my)(0, 0) = 0;
            (*my)(1, 0) = 1;
        }
        else if (rand == 3)
        {
            (*mx)(0, 0) = 1;
            (*mx)(1, 0) = 1;

            (*my)(0, 0) = 0;
            (*my)(1, 0) = 1;
        }

        x_train->push_back(mx);
        y_train->push_back(my);
    }

    // Train model
    model.compile(0.1);
    model.train(x_train, y_train, 1, 8);

    // Create input value
    auto x = make_shared<Matrix>(2, 1);
    (*x)(0, 0) = 1;
    (*x)(1, 0) = 0;

    // Make a prediction
    auto y = model.predict(x);
    cout << *y;

    return 0;
}
