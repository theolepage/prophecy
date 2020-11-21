#pragma once

#include "layer/layer.hh"
#include "tensor/tensor.hh"

namespace prophecy
{
    template <typename T = float>
    class MaxPooling2DLayer final : public Layer<T>
    {
    public:
        MaxPooling2DLayer(const std::vector<uint>& kernel_shape,
                        const uint padding,
                        const uint stride)
            : Layer<T>()
            , kernel_shape_(std::make_shared<std::vector<uint>>(kernel_shape))
            , padding_(padding)
            , stride_(stride)
        {}

        MaxPooling2DLayer(const std::vector<uint>& kernel_shape)
            : Layer<T>()
            , kernel_shape_(std::make_shared<std::vector<uint>>(kernel_shape))
            , padding_(0)
            , stride_(1)
        {}

        virtual ~MaxPooling2DLayer() = default;

        void compile(std::weak_ptr<Layer<T>> prev, std::shared_ptr<Layer<T>> next)
        {
            // Determine output shape
            const auto prev_shape = prev.lock()->get_out_shape();
            const uint c = prev_shape[0];
            const uint h = 1 + (prev_shape[1] + 2 * this->padding_ - this->kernel_shape_->at(0)) / this->stride_;
            const uint w = 1 + (prev_shape[2] + 2 * this->padding_ - this->kernel_shape_->at(1)) / this->stride_;
            std::vector<uint> out_shape({ c, h, w });
            this->out_shape_ = std::make_shared<std::vector<uint>>(out_shape);

            this->compiled_ = true;
            this->prev_ = prev;
            this->next_ = next;
        }

        Tensor<T> feedforward(const Tensor<T>& input, bool training)
        {
            const uint channels = input.get_shape()[0];
            const uint height = input.get_shape()[1];
            const uint width = input.get_shape()[2];

            const uint kernel_height = this->kernel_shape_->at(0);
            const uint kernel_width = this->kernel_shape_->at(1);

            const uint out_rows = (height + 2 * this->padding_ - kernel_height) / this->stride_ + 1;
            const uint out_cols = (width + 2 * this->padding_ - kernel_width) / this->stride_ + 1;
            Tensor<T> out({ channels, out_rows, out_cols });

            std::vector<Coord3D> back;
            for (uint c = 0; c < channels; c++)
            {
                for (uint i = 0; i < out_rows; i++)
                {
                    for (uint j = 0; j < out_cols; j++)
                    {
                        T maximum = static_cast<T>(-1.0 * 10000.0);
                        Coord3D maximum_coords = Coord3D(c, i, j);
                        for (uint k = 0; k < kernel_height * kernel_width; k++)
                        {
                            const int y = i * this->stride_ - this->padding_ + (k / kernel_width);
                            const int x = j * this->stride_ - this->padding_ + (k % kernel_width);

                            if (y < 0 || static_cast<uint>(y) >= height || x < 0 || static_cast<uint>(x) >= width)
                                continue;

                            T value = input({ c, static_cast<uint>(y), static_cast<uint>(x) });
                            if (value >= maximum)
                            {
                                maximum = value;
                                maximum_coords = Coord3D(c, static_cast<uint>(y), static_cast<uint>(x));
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
            uint k = 0;

            for (uint c = 0; c < delta.get_shape()[0]; c++)
                for (uint i = 0; i < delta.get_shape()[1]; i++)
                    for (uint j = 0; j < delta.get_shape()[2]; j++)
                        new_delta(this->mask_indices_->at(k++).to_list()) = delta({ c, i, j });

            prev->backpropagation(new_delta);
        }

    private:
        class Coord3D
        {
        public:
            Coord3D()
            {}

            Coord3D(const uint x, const uint y, const uint z)
                : x_(x), y_(y), z_(z)
            {}

            std::vector<uint> to_list()
            {
                return { x_, y_, z_ };
            }
        private:
            uint x_, y_, z_;
        };

        std::shared_ptr<std::vector<uint>> kernel_shape_;
        const uint padding_;
        const uint stride_;
        std::shared_ptr<std::vector<Coord3D>> mask_indices_;
    };
}