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

    Matrix predict(Matrix x);

    void train(std::vector<Matrix> x,
            std::vector<Matrix y,
            unsigned epoch,
            unsigned batch_size);

    double evaluate(std::vector<Matrix> x, std::vector<Matrix> y);

    void summary();

    void save(std::string path);

    void load(std::string path);

    void compile(double learning_rate);

private:
    std::vector<Layer> layers_;
    bool compiled_;
};
