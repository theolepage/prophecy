#pragma once

#include <string>
#include <vector>

#include "dense_layer.hh"
#include "input_layer.hh"

class Model
{
public:
    Model();

    Model& add(Layer layer);

    void compile(double learning_rate);

    void train(std::vector<Matrix> x,
            std::vector<Matrix y,
            unsigned epoch,
            unsigned batch_size);

    Matrix predict(Matrix x);

    double evaluate(std::vector<Matrix> x, std::vector<Matrix> y);

    void summary();

    void save(std::string path);

    void load(std::string path);

private:
    bool compiled_;
    std::vector<Layer> layers_;
    double learning_rate_;
};
