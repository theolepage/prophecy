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
    double evaluate(std::vector<Matrix> x, std::vector<Matrix> y);

    void compile(double learning_rate);
    void train(std::vector<Matrix> x,
               std::vector<Matrix> y,
               unsigned epochs,
               unsigned batch_size);

    void summary();
    void save(const std::string& path);
    void load(const std::string& path);

private:
    bool compiled_;
    std::vector<Layer> layers_;
    double learning_rate_;
};
