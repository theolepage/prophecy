#pragma once

#include <memory>

#include "../matrix/matrix.hh"

template <typename T>
class Layer
{
public:
    Layer(int nb_neurons)
    : nb_neurons_(nb_neurons)
    {}

    virtual ~Layer() = default;

    virtual Matrix<T> feedforward(const Matrix<T>& input, bool training) = 0;

    virtual void backpropagation(const Matrix<T>* const y) = 0;

    virtual void compile(std::weak_ptr<Layer<T>> prev,
                            std::shared_ptr<Layer<T>> next)
    {
        compiled_ = true;
        prev_ = prev;
        next_ = next;
    }

    int get_nb_neurons(void) const { return nb_neurons_; }
    Matrix<T>& get_delta(void) { return delta_; }
    Matrix<T>& get_last_a(void) { return last_a_; }

protected:
    bool compiled_;
    int nb_neurons_;

    Matrix<T> last_a_;
    Matrix<T> last_z_;
    Matrix<T> delta_;

    std::weak_ptr<Layer<T>> prev_;
    std::shared_ptr<Layer<T>> next_;
};
