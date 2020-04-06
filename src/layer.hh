#pragma once

#include <memory>

#include "matrix.hh"

class Layer
{
public:
    Layer(unsigned nb_neurons);

    unsigned get_nb_neurons();
    std::shared_ptr<Matrix> get_last_a();
    std::shared_ptr<Matrix> get_last_z();
    std::shared_ptr<Matrix> get_delta();

    virtual std::shared_ptr<Matrix> feedforward(std::shared_ptr<Matrix> input, bool training);

    virtual void backpropagation(std::shared_ptr<Matrix> y);

    virtual void compile(std::shared_ptr<Layer> prev,
                         std::shared_ptr<Layer> next);

protected:
    bool compiled_;
    unsigned nb_neurons_;

    std::shared_ptr<Matrix> last_a_;
    std::shared_ptr<Matrix> last_z_;
    std::shared_ptr<Matrix> delta_;

    std::shared_ptr<Layer> prev_;
    std::shared_ptr<Layer> next_;
};
