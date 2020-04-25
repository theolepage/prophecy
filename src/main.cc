#include <iostream>
#include <memory>

#include "tensor/tensor.hh"
#include "model/model.hh"
#include "layer_implem/dense_layer.hh"
#include "layer_implem/conv_2d_layer.hh"
#include "layer_implem/max_pooling_2d_layer.hh"
#include "layer_implem/flatten_layer.hh"
#include "dataset_handler/dataset_handler.hh"

using model_type = float;

static void xor_example(void)
{
    Model<model_type> model = Model<model_type>();
    SigmoidActivationFunction s = SigmoidActivationFunction<model_type>();

    // Create model
    model.add(new InputLayer<model_type>({ 2 }));
    model.add(new DenseLayer<model_type>({ 2 }, s));
    model.add(new DenseLayer<model_type>({ 1 }, s));

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
}

static void cifar_10_example(void)
{
    Model<model_type> model = Model<model_type>();
    SigmoidActivationFunction s = SigmoidActivationFunction<model_type>();
    DatasetHandler dh;
    dh.set_limit(10);
    dh.read("datasets/cifar-10-batches-bin/data_batch_1.bin", set_type::CIFAR_10);

    // Create model
    model.add(new InputLayer<model_type>({ 3, 32, 32 }));

    model.add(new Conv2DLayer<model_type>({ 32, 3, 3, 3 }, s));
    model.add(new Conv2DLayer<model_type>({ 64, 32, 3, 3 }, s));
    model.add(new MaxPooling2DLayer<model_type>({ 2, 2 }, 0, 2));

    model.add(new FlattenLayer<model_type>({ 588, 1 }));
    model.add(new DenseLayer<model_type>({ 128 }, s));
    model.add(new DenseLayer<model_type>({ 10 }, s));

    // Create dataset
    auto x_train = dh.normalize<model_type>(dh.get_training(), static_cast<model_type>(255));
    auto y_train = dh.normalize<model_type>(dh.get_labels(), static_cast<model_type>(1)); // Just to go from byte to float

    // Train model
    model.compile(0.1);
    model.train(x_train, y_train, 1, 1);

    auto res = model.predict(dh.get_training().at(0));
    std::cout << res;
}

int main(void)
{
    switch(1) {
        case 0:
            xor_example();
            break;
        case 1:
            cifar_10_example();
            break;
        default:
            xor_example();
    }
    return 0;
}