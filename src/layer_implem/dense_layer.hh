#pragma once

#include "../layer/hidden_layer.hh"
#include "../matrix/matrix.hh"
#include "../activation_function/activation_function.hh"

template <typename T>
class DenseLayer : public HiddenLayer<T>
{
public:
    DenseLayer(int nb_neurons,
               const ActivationFunction<T>& activation)
    : HiddenLayer<T>(nb_neurons, activation)
    {}

    virtual ~DenseLayer() = default;

    Matrix<T> feedforward(const Matrix<T>& input, bool training)
    {
        auto z = Matrix<T>::dot(this->weights_, input);
        z += this->biases_;
        auto a = z.map(this->activation_.f_);

        if (training)
        {
            this->last_a_ = a;
            this->last_z_ = z;
        }

        if (this->next_ == nullptr)
            return a;
        return this->next_->feedforward(a, training);
    }

    virtual void backpropagation(const Matrix<T>* const y)
    {
        if (this->prev_.expired()) // If we reach InputLayer
            return;

        auto next = std::dynamic_pointer_cast<HiddenLayer<T>>(this->next_);

        if (y != nullptr)
        {
            this->delta_ = Matrix<T>::multiply(
                (this->last_a_ - *y), this->last_z_.map(this->activation_.fd_));
        }
        else
        {
            this->delta_ = Matrix<T>::multiply(
                Matrix<T>::dot(next->get_weights(), next->get_delta(), transpose::LEFT),
                    this->last_z_.map(this->activation_.fd_));
        }

        this->delta_biases_ += this->delta_;
        this->delta_weights_ += Matrix<T>::dot(
                                    this->delta_, this->prev_.lock()->get_last_a(), transpose::RIGHT);
        this->prev_.lock()->backpropagation(nullptr);
    }

    void update(T learning_rate)
    {
        // Update weights_ and biases_
        for (int i = 0; i < this->weights_.get_rows(); i++)
        {
            for (int j = 0; j < this->weights_.get_cols(); j++)
                this->weights_(i, j) -= learning_rate * this->delta_weights_(i, j);
            this->biases_(i, 0) -= learning_rate * this->delta_biases_(i, 0);
        }

        // Reset delta_weights_ and delta_biases_
        this->delta_weights_.fill(fill_type::ZERO);
        this->delta_biases_.fill(fill_type::ZERO);
    }

    void compile(std::weak_ptr<Layer<T>> prev,
                 std::shared_ptr<Layer<T>> next)
    {
        // Initialize weights and biases
        this->weights_ = Matrix<T>(this->nb_neurons_, prev.lock()->get_nb_neurons());
        this->biases_ = Matrix<T>(this->nb_neurons_, 1);
        this->weights_.fill(fill_type::RANDOM_FLOAT);
        this->biases_.fill(fill_type::RANDOM_FLOAT);

        // Initialize delta_weights_ and delta_biases_
        this->delta_weights_ = Matrix<T>(this->nb_neurons_, prev.lock()->get_nb_neurons());
        this->delta_biases_ = Matrix<T>(this->nb_neurons_, 1);

        this->compiled_ = true;
        this->prev_ = prev;
        this->next_ = next;
    }

};