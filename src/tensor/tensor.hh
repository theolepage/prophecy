#pragma once

#include <ostream>
#include <memory>
#include <functional>
#include <cassert>
#include <cstdarg>
#include <algorithm>
#include <iostream>

#include "utils/utils.hh"

enum class fill_type
{
    RANDOM,
    SEQUENCE,
    ZEROS,
    ONES
};

enum class transpose
{
    LEFT,
    RIGHT,
    NO_IMPLICIT
};

template <typename T = float>
class Tensor
{
public:

    /**
    * Constructors
    */

    Tensor()
        : size_(0), shape_(0), data_(nullptr), transposed_(false)
    {}

    Tensor(const std::vector<uint>& shape)
    {
        shape_ = std::make_shared<std::vector<uint>>(shape);
        size_ = compute_size(shape_);
        data_ = std::shared_ptr<T[]>(new T[size_]);
        offset_ = 0L;
        transposed_ = false;
    }

    Tensor(const Tensor<T>& t)
    {
        size_ = t.size_;
        shape_ = t.shape_;
        data_ = t.data_;
        offset_ = 0L;
        transposed_ = t.transposed_;
    }

    Tensor& operator=(const Tensor<T>& t)
    {
        size_ = t.size_;
        shape_ = t.shape_;
        data_ = t.data_;
        offset_ = 0L;
        transposed_ = t.transposed_;
        return *this;
    }

    virtual ~Tensor() = default;

    bool operator==(const Tensor<T>& t)
    {
        if (*shape_ != t.get_shape())
            return false;
        for (int i = 0; i < size_; ++i)
        {
            if (data_[i] != t.data_[i])
                return false;
        }
        return true;
    }

    /**
    * Getters / setters
    */

    std::vector<uint> get_shape() const
    {
        return *shape_;
    }

    int get_size(void) const
    {
        return size_;
    }

    template<typename ...Ts>
    T operator()(Ts&& ...coords) const
    {
        const std::vector<uint> vec = {coords ...};
        return (*this)(vec);
    }

    template<typename ...Ts>
    T& operator()(Ts&& ...coords)
    {
        const std::vector<uint> vec = {coords ...};
        return (*this)(vec);
    }

    T operator()(const std::vector<uint>& coords) const
    {
        return data_[coord_to_index(coords)];
    }

    T& operator()(const std::vector<uint>& coords)
    {
        return data_[coord_to_index(coords)];
    }

    Tensor<T>& reshape(std::vector<uint> shape)
    {
        auto new_shape = std::make_shared<std::vector<uint>>(shape);
        if (compute_size(new_shape) != compute_size(shape_))
            throw "New shape must be of same size.";

        shape_ = new_shape;
        return *this;
    }

    /**
    * Operations (fill, map, op)
    */

    template <typename FUNCTOR_TYPE>
    void fill(FUNCTOR_TYPE& func)
    {
        assert(data_ != nullptr);
        for (int i = offset_; i < size_; ++i)
            data_[i] = func();
    }

    void fill(std::function<T(void)> value_initializer)
    {
        assert(data_ != nullptr);

        for (long long i = offset_; i < size_; i++)
            data_[i] = value_initializer();
    }

    void fill(T value)
    {
        assert(data_ != nullptr);

        for (long long i = offset_; i < size_; i++)
            data_[i] = value;
    }

    void fill(fill_type type)
    {
        assert(data_ != nullptr);

        switch(type)
        {
            case fill_type::RANDOM:
                for (long long i = offset_; i < size_; i++)
                    data_[i] = get_random_float();
                break;
            case fill_type::SEQUENCE:
                for (long long i = offset_; i < size_; i++)
                    data_[i] = i;
                break;
            case fill_type::ZEROS:
                for (long long i = offset_; i < size_; i++)
                    data_[i] = static_cast<T>(0);
                break;
            case fill_type::ONES:
                for (long long i = offset_; i < size_; i++)
                    data_[i] = static_cast<T>(1);
                break;
        }
    }

    Tensor<T>& map_inplace(std::function<T(T)> value_initializer)
    {
        assert(data_ != nullptr);

        for (long long i = offset_; i < size_; i++)
            data_[i] = value_initializer(data_[i]);
        return *this;
    }

    Tensor<T> map(std::function<T(T)> value_initializer) const
    {
        assert(data_ != nullptr);

        Tensor<T> res(*shape_);
        for (long long i = offset_; i < size_; i++)
            res.data_[i] = value_initializer(data_[i]);
        return res;
    }

    Tensor<T> op(const Tensor &right, std::function<T(T, T)> fn) const
    {
        assert(data_ != nullptr);
        if (*shape_ != *right.shape_)
            throw std::invalid_argument("Operations requires the two tensors to have the same shape.");

        Tensor<T> res(*shape_);
        for (long long i = offset_; i < size_; i++)
            res.data_[i] = fn(data_[i], right.data_[i]);
        return res;
    }

    Tensor<T>& op_inplace(const Tensor &right, std::function<T(T, T)> fn)
    {
        assert(data_ != nullptr);
        if (*shape_ != *right.shape_)
            throw std::invalid_argument("Operations requires the two tensors to have the same shape.");

        for (long long i = offset_; i < size_; i++)
            data_[i] = fn(data_[i], right.data_[i]);
        return *this;
    }

    Tensor<T>& op_inplace(T val, std::function<T(T, T)> fn)
    {
        assert(data_ != nullptr);

        for (int i = offset_; i < size_; i++)
            data_[i] = fn(data_[i], val);
        return *this;
    }

    Tensor<T>& operator+=(const Tensor &right)
    {
        return op_inplace(right, [](const T a, const T b) { return a + b; });
    }

    Tensor<T>& operator+=(const T v)
    {
        return op_inplace(v, [](const T a, const  T b) { return a + b; });
    }

    Tensor<T> operator+(const Tensor &right) const
    {
        return op(right, [](const T a, const T b) { return a + b; });
    }

    Tensor<T>& operator-=(const Tensor &right)
    {
        return op_inplace(right, [](const T a, const T b) { return a - b; });
    }

    Tensor<T>& operator-=(const T v)
    {
        return op_inplace(v, [](const T a, const  T b) { return a - b; });
    }

    Tensor<T> operator-(const Tensor &right) const
    {
        return op(right, [](const T a, const  T b) { return a - b; });
    }

    Tensor<T>& operator*=(const Tensor &right)
    {
        return op_inplace(right, [](const T a, const T b) { return a * b; });
    }

    Tensor<T>& operator*=(const T v)
    {
        return op_inplace(v, [](const T a, const  T b) { return a * b; });
    }

    Tensor<T> operator*(const Tensor &right) const
    {
        return op(right, [](const T a, const T b) { return a * b; });
    }

    Tensor<T>& operator/=(const Tensor &right)
    {
        return op_inplace(right, [](const T a, const T b) { return a / b; });
    }

    Tensor<T>& operator/=(const T v)
    {
        return op_inplace(v, [](const T a, const  T b) { return a / b; });
    }

    Tensor<T> operator/(const Tensor &right) const
    {
        return op(right, [](const T a, const T b) { return a / b; });
    }

    Tensor<T> extract(std::vector<uint> coords) const
    {
        auto prevdim = coords.size();
        auto shape_size = shape_->size();
        assert(prevdim <= shape_size);

        while (coords.size() < shape_size)
            coords.push_back(0);

        long long begin = coord_to_index(coords);

        std::vector<uint> new_shape(0);
        for (auto i = prevdim; i < shape_size; i++)
            new_shape.push_back(shape_->at(i));
        if (!new_shape.size())
            new_shape.push_back(1);

        Tensor<T> res(new_shape, data_);
        res.offset_ = begin;
        return res;
    }

    Tensor<T> sum()
    {
        return reduce(0, [](T a, T b) { return a + b; });
    }

    Tensor<T> sum(std::vector<uint> axis)
    {
        return reduce(axis, 0, [](T a, T b) { return a + b; });
    }

    Tensor<T> sum(T subtotal_default)
    {
        return reduce(subtotal_default, [](T a, T b) { return a + b; });
    }

    Tensor<T> reduce(T subtotal_default, std::function<T(T, T)> fn) const
    {
        Tensor<T> res = *this;
        for (int i = shape_->size() - 1; i >= 0; i--)
            res = res.reduce(i, subtotal_default, fn);
        return res;
    }

    Tensor<T> reduce(std::vector<uint> axis, T subtotal_default, std::function<T(T, T)> fn) const
    {
        Tensor<T> res = *this;

        // Remove duplicates and sort in descending order
        std::sort(axis.begin(), axis.end(), std::greater<uint>());
        axis.erase(std::unique(axis.begin(), axis.end()), axis.end());

        for (int dim : axis)
            res = res.reduce(dim, subtotal_default, fn);
        return res;
    }

    Tensor<T> reduce(unsigned dim, T subtotal_default, std::function<T(T, T)> fn)
    {
        // Compute output shape
        std::vector<uint> output_shape;
        for (unsigned i = 0; i < shape_->size(); i++)
            if (i != dim)
                output_shape.push_back(shape_->at(i));
        if (output_shape.size() == 0)
            output_shape = { 1 };
        Tensor<T> res(output_shape);
        int index = 0;

        // Compute the step
        long long step = 1;
        for (unsigned i = dim + 1; i < shape_->size(); i++)
            step *= shape_->at(i);

        // For every element of this dimension, sum what is inside
        long long size_prevdim = 1;
        for (unsigned i = 0; i < dim; i++)
            size_prevdim *= shape_->at(i);
        for (long long i = 0; i < size_prevdim; i++)
            reduce_aux(res, i, step, dim, index, subtotal_default, fn);

        return res;
    }

    void reduce_aux(const Tensor<T>& res,
                    int i,
                    int step,
                    int dim,
                    int& index,
                    T subtotal_default,
                    std::function<T(T, T)> fn) const
    {
        int size_dim = shape_->at(dim);
        long long begin = i * (step * size_dim);
        for (long long i = 0; i < step; i++)
        {
            T subtotal = subtotal_default;
            for (int j = 0; j < size_dim; j++)
                subtotal = fn(subtotal, data_[begin + j * step + i]);
            res.data_[index] = subtotal;
            index += 1;
        }
    }

    /**
    * Matrix operations (transpose, matmul)
    */

    Tensor<T> im2col(const uint kernel_height, const uint kernel_width, const uint padding, const uint stride)
    {
        const uint img_channels = this->shape_->at(0);
        const uint img_height = this->shape_->at(1);
        const uint img_width = this->shape_->at(2);

        const uint kernel_vertical_shifts = 1 + (img_height + 2 * padding - kernel_height) / stride;
        const uint kernel_horizontal_shifts = 1 + (img_width + 2 * padding - kernel_width) / stride;
        const uint patch = img_channels * kernel_height * kernel_width;

        Tensor<T> res({ patch, kernel_vertical_shifts * kernel_horizontal_shifts });

        for (uint c = 0; c < patch; c++)
        {
            const uint vertical_offset = (c / kernel_width) % kernel_height;
            const uint kernel_offset = c % kernel_width;
            const uint channel = c / (kernel_height * kernel_width);

            for (uint i = 0; i < kernel_vertical_shifts; i++)
            {
                for (uint j = 0; j < kernel_horizontal_shifts; j++)
                {
                    int y = i * stride - padding + vertical_offset;
                    int x = j * stride - padding + kernel_offset;

                    T value = 0;
                    if (y >= 0 && static_cast<uint>(y) < img_height && x >= 0 && static_cast<uint>(x) < img_width)
                        value = (*this)({ channel, static_cast<uint>(y), static_cast<uint>(x) });
                    res({ c, i * kernel_horizontal_shifts + j }) = value;
                }
            }
        }

        return res;
    }

    Tensor<T> col2im(std::vector<uint> img_shape, const uint kernel_height, const uint kernel_width, const uint padding, const uint stride)
    {
        const uint img_channels = img_shape[0];
        const uint img_height = img_shape[1];
        const uint img_width = img_shape[2];

        const uint kernel_vertical_shifts = 1 + (img_height + 2 * padding - kernel_height) / stride;
        const uint kernel_horizontal_shifts = 1 + (img_width + 2 * padding - kernel_width) / stride;
        const uint patch = img_channels * kernel_height * kernel_width;

        Tensor<T> res(img_shape);

        for (uint c = 0; c < patch; c++)
        {
            const uint vertical_offset = (c / kernel_width) % kernel_height;
            const uint kernel_offset = c % kernel_width;
            const uint channel = c / (kernel_height * kernel_width);

            for (uint i = 0; i < kernel_vertical_shifts; i++)
            {
                for (uint j = 0; j < kernel_horizontal_shifts; j++)
                {
                    int y = i * stride - padding + vertical_offset;
                    int x = j * stride - padding + kernel_offset;

                    if (y >= 0 && static_cast<uint>(y) < img_height && x >= 0 && static_cast<uint>(x) < img_width)
                        res({ channel, static_cast<uint>(y), static_cast<uint>(x) }) += (*this)({ c, i * kernel_horizontal_shifts + j });
                }
            }
        }

        return res;
    }

    Tensor<T> transpose() const
    {
        assert(data_ != nullptr);
        if (shape_->size() != 2)
            throw "Invalid shape for matrix transpose.";

        Tensor<T> res({shape_->at(1), shape_->at(0)});
        const uint rows = shape_->at(0);
        const uint cols = shape_->at(1);

        for (uint i = 0; i < rows; i++)
            for (uint j = 0; j < cols; j++)
                res.data_[offset_ + j * rows + i] = data_[offset_ + i * cols + j];

        return res;
    }

    Tensor<T> matmul(const Tensor<T>& right) const
    {
        assert(data_ != nullptr);
        if (shape_->size() != 2)
            throw "Invalid shape for matrix multiplication.";

        const uint l_rows = shape_->at(0);
        const uint l_cols = shape_->at(1);
        const uint r_rows = right.shape_->at(0);
        const uint r_cols = right.shape_->at(1);

        if (r_rows != l_cols)
            throw std::invalid_argument("Invalid shapes for matrix multiplication.");

        Tensor<T> res({l_rows, r_cols});
        for (uint r = 0; r < l_rows; r++)
        {
            for (uint c = 0; c < r_cols; c++)
            {
                T tmp = 0;
                for (uint k = 0; k < l_cols; k++)
                    tmp += data_[r * l_cols + k] * right.data_[k * r_cols + c];
                res.data_[r * r_cols + c] = tmp;
            }
        }
        return res;
    }

    /**
    * Miscellaneous (print)
    */

    friend std::ostream& operator<<(std::ostream& os, Tensor& t)
    {
        std::vector<uint> shape = t.get_shape();
        std::vector<uint> coord(shape.size());
        int l = shape.size() - 1;

        int new_line = shape.size();
        for (long long i = 0; i < t.size_; i++)
        {
            // Opening
            if (i != 0 && new_line > 0)
            {
                for (int i = 0; i < new_line; i++)
                    os << std::endl;
                for (int i = 0; i < l - new_line + 1; i++)
                    os << " ";
            }
            while (new_line > 0)
            {
                os << "[";
                new_line--;
            }

            os << " " << t(coord);
            coord[l] += 1;

            // Closing
            for (int j = coord.size() - 1; j >= 0; j--)
            {
                if (coord[j] == shape[j])
                {
                    new_line++;
                    coord[j] = 0;
                    if (j > 0)
                        coord[j - 1]++;
                    os << (j == l ? " ]" : "]");
                }
            }
        }
        os << std::endl;

        return os;
    }

private:
    long long size_;
    std::shared_ptr<std::vector<uint>> shape_;
    std::shared_ptr<T[]> data_;
    long long offset_;
    bool transposed_;

    Tensor(std::vector<uint> shape, std::shared_ptr<T[]> data)
    {
        shape_ = std::make_shared<std::vector<uint>>(shape);
        size_ = compute_size(shape_);
        data_ = data;
        offset_ = 0;
        transposed_ = false;
    }

    long long compute_size(std::shared_ptr<std::vector<uint>> shape)
    {
        long long res = 1;
        for (auto v : *shape)
            res *= v;
        return res;
    }

    long long coord_to_index(std::vector<uint> coords) const
    {
        assert(coords.size() == shape_->size());
        for (unsigned i = 0; i < shape_->size(); i++)
            assert(coords[i] < shape_->at(i));

        long long res = 0;
        long long step = 1;

        for (int dim = shape_->size() - 1; dim >= 0; dim--)
        {
            res += coords[dim] * step;
            step *= shape_->at(dim);
        }

        return res + offset_;
    }

    float get_random_float(void) const
    {
        static constexpr float min = -1.0f;
        static constexpr float max = 1.0f;

        float d = static_cast<float>(RAND_MAX / (max - min));

        return min + static_cast<float>(rand() / d);
    }
};
