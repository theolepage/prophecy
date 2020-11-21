#include <iostream>
#include <memory>

#include "kernel.cuh"
#include "tensor/tensor.hh"
#include "model/model.hh"
#include "layer/dense_layer.hh"
#include "layer/conv_2d_layer.hh"
#include "layer/max_pooling_2d_layer.hh"
#include "layer/flatten_layer.hh"
#include "dataset_handler/dataset_handler.hh"

using model_type = float;

static void xor_example(void)
{
    Model model = Model();
    SigmoidActivationFunction s = SigmoidActivationFunction();

    // Create model
    model.add(InputLayer({ 2 }));
    model.add(DenseLayer(2, s));
    model.add(DenseLayer(1, s));

    // Create dataset
    DatasetHandler dh;
    dh.read("", set_type::XOR);

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
    ReLUActivationFunction r = ReLUActivationFunction<model_type>();
    DatasetHandler dh;
    dh.set_limit(100);
    dh.read("datasets/cifar-10-batches-bin/data_batch_1.bin", set_type::CIFAR_10);

    // Create model
    model.add(InputLayer<model_type>({ 3, 32, 32 }));

    model.add(Conv2DLayer<model_type>(32, { 3, 3 }, r));
    model.add(Conv2DLayer<model_type>(64, { 3, 3 }, r));
    model.add(MaxPooling2DLayer<model_type>({ 2, 2 }, 0, 2));

    model.add(FlattenLayer<model_type>());
    model.add(DenseLayer<model_type>(128, s));
    model.add(DenseLayer<model_type>(10, s));

    // Create dataset
    auto x_train = dh.get_training();
    auto y_train = dh.get_labels();
    // auto x_train = dh.normalize<model_type>(dh.get_training(), static_cast<model_type>(255));
    // auto y_train = dh.normalize<model_type>(dh.get_labels(), static_cast<model_type>(1)); // Just to go from byte to float

    // Train model
    model.compile(0.1);
    model.train(x_train, y_train, 10, 128);

    // Evaluate model
    auto res = model.predict(x_train.at(0));
    std::cout << x_train[0];
    std::cout << res;
    std::cout << y_train[0];
}

static void mnist_example(void)
{
    Model<model_type> model = Model<model_type>();
    SigmoidActivationFunction s = SigmoidActivationFunction<model_type>();
    ReLUActivationFunction r = ReLUActivationFunction<model_type>();
    DatasetHandler dh;
    dh.set_limit(6000);
    dh.read("datasets/mnist/", set_type::MNIST);

    // Create model
    // model.add(InputLayer<model_type>({ 1, 28, 28 }));
    // model.add(FlattenLayer<model_type>());
    // model.add(DenseLayer<model_type>({ 32 }, r));
    // model.add(DenseLayer<model_type>({ 10 }, r));

    model.add(InputLayer<model_type>({ 1, 28, 28 }));
    model.add(Conv2DLayer<model_type>(32, { 3, 3 }, s));
    model.add(MaxPooling2DLayer<model_type>({ 2, 2 }, 0, 2));
    model.add(FlattenLayer<model_type>());
    model.add(DenseLayer<model_type>(100, s));
    model.add(DenseLayer<model_type>(10, s));

    // Create dataset
    auto x_train = dh.get_training();
    auto y_train = dh.get_labels();

    // Train model
    model.compile(0.01);
    model.train(x_train, y_train, 10, 32);

    auto res = model.predict(x_train[0]);
    std::cout << x_train[0];
    std::cout << res;
    std::cout << y_train[0];

    res = model.predict(x_train[1]);
    std::cout << x_train[1];
    std::cout << res;
    std::cout << y_train[1];

    res = model.predict(x_train[2]);
    std::cout << x_train[2];
    std::cout << res;
    std::cout << y_train[2];
}

int main(void)
{
    #ifdef CUDA_ENABLED
    kernel();
    #endif

    switch(0) {
        case 0:
            xor_example();
            break;
        case 1:
            cifar_10_example();
            break;
        case 2:
            mnist_example();
            break;
        default:
            xor_example();
    }
    return 0;
}