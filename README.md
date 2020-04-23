# prophecy

Check this out Google.

## Usage

```cpp
#include <iostream>
#include <memory>

#include "model/model.hh"
#include "layer_implem/dense_layer.hh"
#include "dataset_handler/dataset_handler.hh"
#include "tensor/tensor.hh"

using model_type = float;


int main(void)
{
    Model<model_type> model = Model<model_type>();
    SigmoidActivationFunction s = SigmoidActivationFunction<model_type>();

    // Create model
    model.add(new InputLayer<model_type>(2));
    model.add(new DenseLayer<model_type>(2, s));
    model.add(new DenseLayer<model_type>(1, s));

    // Create dataset
    Dataset_handler d_s;
    d_s.read(nullptr, set_type::XOR);

    // Train model
    model.compile(0.1);
    model.train(d_s.get_training(), d_s.get_labels(), 10000, 1);

    // Test the model
    for (size_t i = 0; i < d_s.get_training().size(); i++)
    {
        auto x = d_s.get_training().at(i);
        auto x_t = x.transpose();
        auto y = model.predict(x);
        std::cout << "Input:  " << x_t;
        std::cout << "Output: " << y << std::endl;
    }

    return 0;
}

```

## To-Do

Refer to [this page](https://github.com/theolepage/prophecy/projects/1).
