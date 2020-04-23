#include <iostream>
#include <memory>

#include "model/model.hh"
#include "layer_implem/dense_layer.hh"
#include "dataset_handler/dataset_handler.hh"
#include "tensor/tensor.hh"

using model_type = float;

void xor_example(void)
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
}
/*
static void cirfa_10_example(void)
{
    Model<model_type> model = Model<model_type>();
    SigmoidActivationFunction s = SigmoidActivationFunction<model_type>();
    DatasetHandler d_s;
    d_s.read("data/data_batch_1.bin", set_type::CIFAR_10);



    // Create model
    model.add(new InputLayer<model_type>(32*32));
    model.add(new DenseLayer<model_type>(50, s));
    model.add(new DenseLayer<model_type>(50, s));
    model.add(new DenseLayer<model_type>(50, s));
    model.add(new DenseLayer<model_type>(10, s));

    // Create dataset
    auto binarized_train = d_s.binarize();
    auto normalized_train = d_s.normalize<model_type>(binarized_train, static_cast<model_type>(255));
    auto normalized_label = d_s.normalize<model_type>(d_s.get_labels(), static_cast<model_type>(1)); // Just to go from byte to float
    //for (auto& tensor : normalized_train)
    //    tensor.flatten_inplace();

    // Train model
    model.compile(0.1);
    model.train(normalized_train, normalized_label, 1, 1);

    for (size_t i = 0; i < normalized_train.size(); i++)
    {
        auto x = normalized_train.at(i);
        auto x_t = x.transpose();
        auto y = model.predict(x);
        auto expected_y = normalized_label.at(i).transpose();
        std::cout << "Output: " << y << std::endl;
        std::cout << "Expected: " << expected_y << std::endl;
    }
}*/

int main(void)
{
    xor_example();
    //cirfa_10_example();

    return 0;
}
