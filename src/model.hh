#pragma once

#include <memory>
#include <string>
#include <vector>

#include "input_layer.hh"
#include "hidden_layer.hh"

class Model
{
public:
    Model();

    Model& add(std::shared_ptr<Layer> layer);

    std::shared_ptr<Matrix> predict(std::shared_ptr<Matrix> x);

    void compile(double learning_rate);
    void train(std::shared_ptr<std::vector<std::shared_ptr<Matrix>>> x,
               std::shared_ptr<std::vector<std::shared_ptr<Matrix>>> y,
               unsigned epochs,
               unsigned batch_size);

    void summary();
    void save(const std::string& path);
    void load(const std::string& path);

private:
    bool compiled_;
    double learning_rate_;
    std::vector<std::shared_ptr<Layer>> layers_;
};
