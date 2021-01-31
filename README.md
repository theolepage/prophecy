# prophecy

Check this out Google.

## Compilation

1. `conan remote add omaralvarez https://api.bintray.com/conan/omaralvarez/public-conan`
2. `mkdir build; cd build;`
3. `conan install .. --build=missing`
4. `cmake ..`
5. `make`

## Usage

```cpp
#include <iostream>
#include <memory>

#include "model/model.hh"
#include "layer/dense_layer.hh"
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
    DatasetHandler dh;
    dh.read(nullptr, set_type::XOR);

    // Train model
    model.compile(0.1);
    model.train(dh.get_training(), dh.get_labels(), 10000, 1);

    // Test the model
    for (size_t i = 0; i < dh.get_training().size(); i++)
    {
        auto x = dh.get_training().at(i);
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
