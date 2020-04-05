#include <iostream>

#include "model.hh"
#include "dense_layer.hh"

using namespace std;

int main(void)
{
    Model model = Model();

    // Create model
    model.add(make_shared<InputLayer>(2));
    model.add(make_shared<DenseLayer>(8, make_shared<SigmoidActivationFunction>()));
    model.add(make_shared<DenseLayer>(2, make_shared<SigmoidActivationFunction>()));

    // Train model
    model.compile(0.01);
    // model.train(x_train, y_train, 10, 8);

    // Create input value
    auto x = make_shared<Matrix>(2, 1);
    (*x)(0, 0) = 1;
    (*x)(1, 0) = 0;

    // Make a prediction
    auto y = model.predict(x);
    cout << *y;

    return 0;
}
