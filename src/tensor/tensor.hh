#pragma once
#include <ostream>
#include <memory>
#include <functional>
#include <cassert>
#include <cstdarg>
#include <algorithm>
#include <iostream>

enum class fill_type
{
    RANDOM,
    SEQUENCE,
    ZEROS,
    ONES
};

template <typename T>
class Tensor
{
public:

    /**
    * Constructors
    */

    Tensor()
        : size_(0), shape_(0), data_(nullptr), prevent_free_(false), transposed_(false)
    {}

    Tensor(const std::vector<int>& shape)
        : shape_(std::make_shared<std::vector<int>>(shape))
    {
        size_ = compute_size(shape);
        data_ = new T[size_];
        prevent_free_ = false;
        transposed_ = false;
    }

    Tensor(const Tensor<T>& t)
    {
        size_ = t.size_;
        shape_ = t.shape_;
        data_ = t.data_;
        prevent_free_ = t.prevent_free_;
        transposed_ = t.transposed_;
    }

    Tensor& operator=(const Tensor<T>& t)
    {
        size_ = t.size_;
        shape_ = t.shape_;
        data_ = t.data_;
        prevent_free_ = t.prevent_free_;
        transposed_ = t.transposed_;
        return *this;
    }

    virtual ~Tensor()
    {
        if (!prevent_free_)
            delete[] data_;
    }

    /**
    * Getters / setters
    */

    std::vector<int>& get_shape()
    {
        return *shape_;
    }

    int get_size() const
    {
        return size_;
    }

    int get_sub_matrix() const
    {
        return prevent_free_;
    }

    T operator()(std::vector<int> coords) const
    {
        return data_[coord_to_index(coords)];
    }

    T& operator()(std::vector<int> coords)
    {
        return data_[coord_to_index(coords)];
    }

    void reshape(std::initializer_list<int> shape)
    {
        if (compute_size(shape) != compute_size(shape_))
            throw "New shape must be of same size.";

        shape_ = shape;
    }

    /**
    * Operations (fill, map, op)
    */

    void fill(std::function<T(void)> value_initializer)
    {
        assert(data_ != nullptr);

        for (long long i = 0; i < size_; i++)
            data_[i] = value_initializer();
    }

    void fill(T value)
    {
        assert(data_ != nullptr);

        for (long long i = 0; i < size_; i++)
            data_[i] = value;
    }

    void fill(fill_type type)
    {
        assert(data_ != nullptr);

        switch(type)
        {
            case fill_type::RANDOM:
                for (long long i = 0; i < size_; i++)
                    data_[i] = get_random_float();
                break;
            case fill_type::SEQUENCE:
                for (long long i = 0; i < size_; i++)
                    data_[i] = i;
                break;
            case fill_type::ZEROS:
                for (long long i = 0; i < size_; i++)
                    data_[i] = static_cast<T>(0);
                break;
            case fill_type::ONES:
                for (long long i = 0; i < size_; i++)
                    data_[i] = static_cast<T>(1);
                break;
        }
    }

    Tensor<T>& map_inplace(std::function<T(T)> value_initializer)
    {
        assert(data_ != nullptr);

        for (long long i = 0; i < size_; i++)
            data_[i] = value_initializer(data_[i]);
        return *this;
    }

    Tensor<T> map(std::function<T(T)> value_initializer) const
    {
        assert(data_ != nullptr);

        Tensor<T> res(shape_);
        for (long long i = 0; i < size_; i++)
            res.data_[i] = value_initializer(data_[i]);
        return res;
    }

    Tensor<T> op(const Tensor &right, std::function<T(T, T)> fn)
    {
        assert(data_ != nullptr);
        if (shape_ != right.shape_)
            throw std::invalid_argument("Operations requires the two tensors to have the same shape.");

        Tensor<T> res(shape_);
        for (long long i = 0; i < size_; i++)
            res.data_[i] = fn(data_[i], right.data_[i]);
        return res;
    }

    Tensor<T>& op_inplace(const Tensor &right, std::function<T(T, T)> fn)
    {
        assert(data_ != nullptr);
        if (shape_ != right.shape_)
            throw std::invalid_argument("Operations requires the two tensors to have the same shape.");

        for (long long i = 0; i < get_size(); i++)
            data_[i] = fn(data_[i], right.data_[i]);
        return *this;
    }

    Tensor<T>& operator+=(const Tensor &right)
    {
        return op_inplace(right, [](T a, T b) { return a + b; });
    }

    Tensor<T> operator+(const Tensor &right)
    {
        return op(right, [](T a, T b) { return a + b; });
    }

    Tensor<T>& operator-=(const Tensor &right)
    {
        return op_inplace(right, [](T a, T b) { return a - b; });
    }

    Tensor<T> operator-(const Tensor &right)
    {
        return op(right, [](T a, T b) { return a - b; });
    }

    Tensor<T>& operator*=(const Tensor &right)
    {
        return op_inplace(right, [](T a, T b) { return a * b; });
    }

    Tensor<T> operator*(const Tensor &right)
    {
        return op(right, [](T a, T b) { return a * b; });
    }

    Tensor<T>& operator/=(const Tensor &right)
    {
        return op_inplace(right, [](T a, T b) { return a / b; });
    }

    Tensor<T> operator/(const Tensor &right)
    {
        return op(right, [](T a, T b) { return a / b; });
    }

    Tensor<T> extract(std::vector<int> coords)
    {
        unsigned prevdim = coords.size();
        unsigned shape_size = shape_->size();
        assert(prevdim <= shape_size);

        while (coords.size() < shape_size)
            coords.push_back(0);

        long long begin = coord_to_index(coords);

        std::vector<int> new_shape(0);
        for (unsigned i = prevdim; i < shape_size; i++)
            new_shape.push_back(shape_->at(i));
        if (!new_shape.size())
            new_shape.push_back(1);

        Tensor<T> res(new_shape, data_ + begin);
        return res;
    }

    Tensor<T> reduce(T subtotal_default, std::function<T(T, T)> fn)
    {
        Tensor<T> res = *this;
        for (int i = shape_->size() - 1; i >= 0; i--)
            res = res.reduce(i, subtotal_default, fn);
        return res;
    }

    Tensor<T> reduce(std::vector<int> axis, T subtotal_default, std::function<T(T, T)> fn)
    {
        Tensor<T> res = *this;

        // Remove duplicates and sort in descending order
        std::sort(axis.begin(), axis.end(), std::greater<int>());
        axis.erase(std::unique(axis.begin(), axis.end()), axis.end());

        int a = 0;
        for (int dim : axis)
        {
            res = res.reduce(a, dim, subtotal_default, fn);
            a++;
        }

        res.prevent_free_ = false;
        return res;
    }

    Tensor<T> reduce(int a, unsigned dim, T subtotal_default, std::function<T(T, T)> fn)
    {
        // Compute output shape
        std::vector<int> output_shape;
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

        res.prevent_free_ = true;
        if (a != 0)
            delete[] data_;

        return res;
    }

    void reduce_aux(const Tensor<T>& res, unsigned i, int step, int dim, int& index, T subtotal_default, std::function<T(T, T)> fn)
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

    Tensor<T> transpose()
    {
        assert(data_ != nullptr);
        if (shape_->size() > 2)
            throw "Invalid shape for matrix transpose.";

        Tensor<T> res(shape_);
        int rows = shape_->at(0);
        int cols = shape_->at(1);

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                res.data_[j * rows + i] = data_[i * cols + j];

        return res;
    }

    Tensor<T> matmul(const Tensor<T>& right)
    {
        assert(data_ != nullptr);
        if (shape_->size() > 2)
            throw "Invalid shape for matrix multiplication.";

        int l_rows = shape_->at(0);
        int l_cols = shape_->at(1);
        int r_rows = right.shape_->at(0);
        int r_cols = right.shape_->at(1);

        if (r_rows != l_cols)
            throw std::invalid_argument("Invalid shapes for matrix multiplication.");

        Tensor<T> res(l_rows, r_cols);
        for (unsigned r = 0; r < l_rows; r++)
        {
            for (unsigned c = 0; c < r_cols; c++)
            {
                T tmp = 0;
                for (unsigned k = 0; k < l_cols; k++)
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
        std::vector<int> shape = t.get_shape();
        std::vector<int> coord(shape.size());
        int l = shape.size() - 1;

        int new_line = shape.size();
        for (long long i = 0; i < t.get_size(); i++)
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
    std::shared_ptr<std::vector<int>> shape_;
    T *data_;
    bool prevent_free_;
    bool transposed_;

    Tensor(std::vector<int> shape, T *data)
        : shape_(std::shared_ptr<std::vector<int>>(shape))
    {
        size_ = compute_size(shape);
        data_ = data;
        prevent_free_ = true;
        transposed_ = false;
    }

    long long compute_size(std::vector<int> shape)
    {
        long long res = 1;
        for (auto v : shape)
            res *= v;
        return res;
    }

    long long coord_to_index(std::vector<int> coords)
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

        return res;
    }

    float get_random_float(void)
    {
        static constexpr float min = -1.0f;
        static constexpr float max = 1.0f;

        float d = static_cast<float>(RAND_MAX / (max - min));

        return min + static_cast <float>(rand() / d);
    }
};
