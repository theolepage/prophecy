#pragma once

#include "../layer/layer.hh"
#include "../tensor/tensor.hh"

template <typename T>
class MaxPooling2DLayer final : public Layer<T>
{
public:
    MaxPooling2DLayer(const std::vector<int>& shape,
                      int padding,
                      int stride)
        : Layer<T>(shape)
        , padding_(padding)
        , stride_(stride)
    {}

    MaxPooling2DLayer(const std::vector<int>& shape)
        : Layer<T>(shape)
        , padding_(0)
        , stride_(1)
    {}

    virtual ~MaxPooling2DLayer() = default;

    Tensor<T> feedforward(const Tensor<T>& input, bool training)
    {
        int channels = input.get_shape()[0];
        int height = input.get_shape()[1];
        int width = input.get_shape()[2];

        int kernel_height = this->shape_->at(0);
        int kernel_width = this->shape_->at(1);

        int out_rows = (height + 2 * this->padding_ - kernel_height) / this->stride_ + 1;
        int out_cols = (width + 2 * this->padding_ - kernel_width) / this->stride_ + 1;
        Tensor<T> out({ channels, out_rows, out_cols });

        std::vector<Coord3D> back;
        for (int c = 0; c < channels; c++)
        {
            for (int i = 0; i < out_rows; i++)
            {
                for (int j = 0; j < out_cols; j++)
                {
                    int maximum = 0;
                    Coord3D maximum_coords = Coord3D();
                    for (int k = 0; k < kernel_height * kernel_width; k++)
                    {
                        int y = i * this->stride_ - this->padding_ + (k / kernel_width);
                        int x = j * this->stride_ - this->padding_ + (k % kernel_width);

                        if (y < 0 || y >= height || x < 0 || x >= width)
                            continue;

                        int value = input({ c, y, x });
                        if (value > maximum)
                        {
                            maximum = value;
                            maximum_coords = Coord3D(c, y, x);
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

        Tensor<T> new_delta(prev->get_shape());
        int k = 0;

        for (int c = 0; c < delta.get_shape()[0]; c++)
            for (int i = 0; i < delta.get_shape()[1]; i++)
                for (int j = 0; j < delta.get_shape()[2]; j++)
                    new_delta(this->mask_indices_->at(k++).to_list()) = delta({ c, i, j });

        prev->backpropagation(new_delta);
    }

private:
    class Coord3D
    {
    public:
        Coord3D()
        {}

        Coord3D(int x, int y, int z)
            : x_(x), y_(y), z_(z)
        {}

        std::vector<int> to_list()
        {
            return { x_, y_, z_ };
        }
    private:
        int x_, y_, z_;
    };

    int padding_;
    int stride_;
    std::shared_ptr<std::vector<Coord3D>> mask_indices_;
};