#pragma once

#include "layer/layer.hh"
#include "tensor/tensor.hh"

template <typename T = float>
class MaxPooling2DLayer final : public Layer<T>
{
public:
    MaxPooling2DLayer(const std::vector<unsigned int>& kernel_shape,
                      const unsigned int padding,
                      const unsigned int stride)
        : Layer<T>()
        , kernel_shape_(std::make_shared<std::vector<unsigned int>>(kernel_shape))
        , padding_(padding)
        , stride_(stride)
    {}

    MaxPooling2DLayer(const std::vector<unsigned int>& kernel_shape)
        : Layer<T>()
        , kernel_shape_(std::make_shared<std::vector<unsigned int>>(kernel_shape))
        , padding_(0)
        , stride_(1)
    {}

    virtual ~MaxPooling2DLayer() = default;

    void compile(std::weak_ptr<Layer<T>> prev, std::shared_ptr<Layer<T>> next)
    {
        // Determine output shape
        const auto prev_shape = prev.lock()->get_out_shape();
        const unsigned int c = prev_shape[0];
        const unsigned int h = 1 + (prev_shape[1] + 2 * this->padding_ - this->kernel_shape_->at(0)) / this->stride_;
        const unsigned int w = 1 + (prev_shape[2] + 2 * this->padding_ - this->kernel_shape_->at(1)) / this->stride_;
        std::vector<unsigned int> out_shape({ c, h, w });
        this->out_shape_ = std::make_shared<std::vector<unsigned int>>(out_shape);

        this->compiled_ = true;
        this->prev_ = prev;
        this->next_ = next;
    }

    Tensor<T> feedforward(const Tensor<T>& input, bool training)
    {
        const unsigned int channels = input.get_shape()[0];
        const unsigned int height = input.get_shape()[1];
        const unsigned int width = input.get_shape()[2];

        const unsigned int kernel_height = this->kernel_shape_->at(0);
        const unsigned int kernel_width = this->kernel_shape_->at(1);

        const unsigned int out_rows = (height + 2 * this->padding_ - kernel_height) / this->stride_ + 1;
        const unsigned int out_cols = (width + 2 * this->padding_ - kernel_width) / this->stride_ + 1;
        Tensor<T> out({ channels, out_rows, out_cols });

        std::vector<Coord3D> back;
        for (unsigned int c = 0; c < channels; c++)
        {
            for (unsigned int i = 0; i < out_rows; i++)
            {
                for (unsigned int j = 0; j < out_cols; j++)
                {
                    T maximum = static_cast<T>(-1.0 * 10000.0);
                    Coord3D maximum_coords = Coord3D(c, i, j);
                    for (unsigned int k = 0; k < kernel_height * kernel_width; k++)
                    {
                        const int y = i * this->stride_ - this->padding_ + (k / kernel_width);
                        const int x = j * this->stride_ - this->padding_ + (k % kernel_width);

                        if (y < 0 || static_cast<unsigned int>(y) >= height || x < 0 || static_cast<unsigned int>(x) >= width)
                            continue;

                        T value = input({ c, static_cast<unsigned int>(y), static_cast<unsigned int>(x) });
                        if (value >= maximum)
                        {
                            maximum = value;
                            maximum_coords = Coord3D(c, static_cast<unsigned int>(y), static_cast<unsigned int>(x));
                        }
                    }
                    out({ c, i, j }) = maximum;
                    back.push_back(maximum_coords);
                }
            }
        }

        if (training)
        {
            this->last_a_ = out;
            this->mask_indices_ = std::make_shared<std::vector<Coord3D>>(back);
        }

        if (this->next_ == nullptr)
            return input;
        return this->next_->feedforward(out, training);
    }

    void backpropagation(Tensor<T>& delta)
    {
        auto prev = this->prev_.lock();

        Tensor<T> new_delta(prev->get_out_shape());
        unsigned int k = 0;

        for (unsigned int c = 0; c < delta.get_shape()[0]; c++)
            for (unsigned int i = 0; i < delta.get_shape()[1]; i++)
                for (unsigned int j = 0; j < delta.get_shape()[2]; j++)
                    new_delta(this->mask_indices_->at(k++).to_list()) = delta({ c, i, j });

        prev->backpropagation(new_delta);
    }

private:
    class Coord3D
    {
    public:
        Coord3D()
        {}

        Coord3D(const unsigned int x, const unsigned int y, const unsigned int z)
            : x_(x), y_(y), z_(z)
        {}

        std::vector<unsigned int> to_list()
        {
            return { x_, y_, z_ };
        }
    private:
        unsigned int x_, y_, z_;
    };

    std::shared_ptr<std::vector<unsigned int>> kernel_shape_;
    const unsigned int padding_;
    const unsigned int stride_;
    std::shared_ptr<std::vector<Coord3D>> mask_indices_;
};