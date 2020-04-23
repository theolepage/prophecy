#pragma once

#include <ostream>
#include <memory>
#include <functional>
#include <cassert>
#include <cstdarg>
#include <algorithm>
#include <any>

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

template <typename T>
class Tensor
{
public:

    /**
    * Constructors
    */

    Tensor()
        : size_(0), shape_(0), data_(nullptr)
    {}

    Tensor(std::vector<int> shape)
        : shape_(shape)
    {
        size_ = compute_size(shape);
        data_ = std::shared_ptr<T[]>(new T[size_]);
    }

    template<typename ...Ts>
    Tensor(Ts... inputs) : shape_{inputs...}
    {
        size_ = compute_size(shape_);
        data_ = std::shared_ptr<T[]>(new T[size_]);
    }

    Tensor(const Tensor<T>& t)
    {
        size_ = t.size_;
        shape_ = t.shape_;
        data_ = t.data_;
    }

    Tensor& operator=(const Tensor<T>& t)
    {
        size_ = t.size_;
        shape_ = t.shape_;
        data_ = t.data_;
        return *this;
    }

    bool operator==(const Tensor<T>& t)
    {
        if (shape_ != t.get_shape())
            return false;
        for (int i = 0; i < size_; ++i)
        {
            if (data_[i] != t.data_[i])
                return false;
        }
        return true;
    }

    virtual ~Tensor() = default;

    /**
    * Getters / setters
    */

    std::vector<int> get_shape(void) const
    {
        return shape_;
    }

    int get_size(void) const
    {
        return size_;
    }

    template<typename ...Ts>
    T operator()(Ts&& ...coords) const
    {
        const std::vector<int> vec = {coords ...};
        return (*this)(vec);
    }

    template<typename ...Ts>
    T& operator()(Ts&& ...coords)
    {
        const std::vector<int> vec = {coords ...};
        return (*this)(vec);
    }

    T operator()(const std::vector<int>& coords) const
    {
        // shape: { 3, 3 }
        // coord: { 2 }
        // out: { 3, 4, 5 }
        return data_[coord_to_index(coords)];
    }

    T& operator()(const std::vector<int>& coords)
    {
        return data_[coord_to_index(coords)];
    }

    void reshape(std::initializer_list<int> shape)
    {
        // To-Do: assert compute_size(shape) == compute_size(shape_)

        size_ = compute_size(shape);
        shape_ = shape;
    }

    /**
    * Operations (fill, map, op)
    */

    template <typename FUNCTOR_TYPE>
    void fill(FUNCTOR_TYPE& func)
    {
        assert(data_ != nullptr);
        for (int i = 0; i < size_; ++i)
            data_[i] = func();
    }

    void fill(std::function<T(void)> value_initializer)
    {
        assert(data_ != nullptr);

        for (int i = 0; i < size_; i++)
            data_[i] = value_initializer();
    }

    void fill(T value)
    {
        assert(data_ != nullptr);

        for (int i = 0; i < size_; i++)
            data_[i] = value;
    }

    void fill(fill_type type)
    {
        assert(data_ != nullptr);

        switch(type)
        {
            case fill_type::RANDOM:
                for (int i = 0; i < size_; i++)
                    data_[i] = get_random_float();
                break;
            case fill_type::SEQUENCE:
                for (int i = 0; i < size_; i++)
                    data_[i] = i;
                break;
            case fill_type::ZEROS:
                for (int i = 0; i < size_; i++)
                    data_[i] = static_cast<T>(0);
                break;
            case fill_type::ONES:
                for (int i = 0; i < size_; i++)
                    data_[i] = static_cast<T>(1);
                break;
        }
    }

    Tensor<T>& map_inplace(std::function<T(T)> value_initializer)
    {
        assert(data_ != nullptr);

        for (int i = 0; i < size_; i++)
            data_[i] = value_initializer(data_[i]);
        return *this;
    }

    Tensor<T> map(std::function<T(T)> value_initializer) const
    {
        assert(data_ != nullptr);

        Tensor<T> res(shape_);
        for (int i = 0; i < size_; i++)
            res.data_[i] = value_initializer(data_[i]);
        return res;
    }

    Tensor<T> op(const Tensor &right, std::function<T(T, T)> fn) const
    {
        assert(data_ != nullptr);
        if (shape_ != right.shape_)
            throw std::invalid_argument("Operations requires the two tensors to have the same shape.");

        Tensor<T> res(shape_);
        for (int i = 0; i < size_; i++)
            res.data_[i] = fn(data_[i], right.data_[i]);
        return res;
    }

    Tensor<T>& op_inplace(const Tensor &right, std::function<T(T, T)> fn)
    {
        assert(data_ != nullptr);
        if (shape_ != right.shape_)
            throw std::invalid_argument("Operations requires the two tensors to have the same shape.");

        for (int i = 0; i < size_; i++)
            data_[i] = fn(data_[i], right.data_[i]);
        return *this;
    }

    Tensor<T>& op_inplace(T val, std::function<T(T, T)> fn)
    {
        assert(data_ != nullptr);

        for (int i = 0; i < size_; i++)
            data_[i] = fn(data_[i], val);
        return *this;
    }

    Tensor<T>& operator+=(const Tensor &right)
    {
        return op_inplace(right, [](const T a, const T b) { return a + b; });
    }

    Tensor<T> operator+(const Tensor &right) const
    {
        return op(right, [](const T a, const T b) { return a + b; });
    }

    Tensor<T>& operator-=(const Tensor &right)
    {
        return op_inplace(right, [](const T a, const T b) { return a - b; });
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

    Tensor<T> operator/(const Tensor &right) const
    {
        return op(right, [](const T a, const T b) { return a / b; });
    }

    Tensor<T> get(std::vector<int> coords) const
    {
        int prevdim = coords.size();
        int shape_size = shape_.size();
        assert(prevdim <= shape_size);

        while (coords.size() < shape_size)
            coords.push_back(0);

        int begin = coord_to_index(coords);

        std::vector<int> new_shape(0);
        for (int i = prevdim; i < shape_size; i++)
            new_shape.push_back(shape_[i]);
        if (!new_shape.size())
            new_shape.push_back(1);

        std::shared_ptr<T[]> sub_data(&(data_[begin]));

        Tensor<T> res(new_shape, sub_data);
        return res;
    }

    Tensor<T> reduce(T subtotal_default, std::function<T(T, T)> fn) const
    {
        Tensor<T> res = *this;
        for (int i = shape_.size() - 1; i >= 0; i--)
            res = res.reduce(i, subtotal_default, fn);
        return res;
    }

    Tensor<T> reduce(std::vector<int> axis, T subtotal_default, std::function<T(T, T)> fn) const
    {
        Tensor<T> res = *this;

        // Remove duplicates and sort in descending order
        std::sort(axis.begin(), axis.end(), std::greater<int>());
        axis.erase(std::unique(axis.begin(), axis.end()), axis.end());

        for (int dim : axis)
            res = res.reduce(dim, subtotal_default, fn);
        return res;
    }

    Tensor<T> reduce(int dim, T subtotal_default, std::function<T(T, T)> fn) const
    {
        // Compute output shape
        std::vector<int> output_shape;
        for (int i = 0; i < shape_.size(); i++)
            if (i != dim)
                output_shape.push_back(shape_[i]);
        if (output_shape.size() == 0)
            output_shape = { 1 };
        Tensor<T> res(output_shape);
        int index = 0;

        // Compute the step
        int step = 1;
        for (int i = dim + 1; i < shape_.size(); i++)
            step *= shape_[i];

        // For every element of this dimension, sum what is inside
        int size_prevdim = 1;
        for (int i = 0; i < dim; i++)
            size_prevdim *= shape_[i];
        for (int i = 0; i < size_prevdim; i++)
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
        int size_dim = shape_[dim];
        int begin = i * (step * size_dim);
        for (int i = 0; i < step; i++)
        {
            T subtotal = subtotal_default;
            for (int j = 0; j < size_dim; j++)
                subtotal = fn(subtotal, data_[begin + j * step + i]);
            res.data_[index] = subtotal;
            index += 1;
        }
    }

    Tensor<T> append(const Tensor<T>& right)
    {
        // shape: { 2, 4, 4 }
        // right: { 1, 4, 4 }
        // output: { 3, 4, 4 }

        // shape: { 1, 2, 4, 4 }
        // right: { 4, 4 }
        // output: { 1, 3, 4, 4 }

        // shape: { 1, 4, 4 }
        // right: { 1, 4, 4 }
        // out: { 2, 4, 4 }
    }

    Tensor<T> conv3D(const Tensor &kernel, int stride) const
    {
        // Checks on dimensions
        assert(shape_.size() == 3 && kernel.shape_.size() == 3)
        assert(shape_[0] == kernel.shape_[0]);

        int padding = 1;

        // Determine output shape
        int out_h = 1 + (shape_[0] + 2 * padding - kernel.shape[0]) / stride;
        int out_w = 1 + (shape_[1] + 2 * padding - kernel.shape[1]) / stride;
        Tensor<T> res({ out_h, out_w });

        // Compute convolution
        for (int c = 0; c < shape_[0]; c++)
            for (int i = 0; i < out_h; i++)
                for (int j = 0; j < out_w; j++)
                    for (int k = 0; k < kernel.shape_[0]; k++)
                        for (int l = 0; l < kernel.shape_[1]; l++)
                            res({ i, j }) += (*this)({ c, stride * i + k, stride * j + l }) * kernel({ c, k, l });
    }

    /**
    * Matrix operations (transpose, matmul)
    */

    Tensor<T> transpose(void) const
    {
        assert(data_ != nullptr);
        assert(shape_.size() == 1 || shape_.size() == 2);

        if (shape_.size() == 1)
            return *this;

        Tensor<T> res(shape_.at(1), shape_.at(0));
        int rows = res.shape_[0];
        int cols = res.shape_[1];

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                res.data_[j * rows + i] = data_[i * cols + j];

        return res;
    }

    Tensor<T> matmul(const Tensor<T>& right) const
    {
        assert(data_ != nullptr);
        assert(shape_.size() == 1 || shape_.size() == 2);

        int l_rows = shape_[0];
        int l_cols = shape_[1];
        int r_rows = right.shape_[0];
        int r_cols = right.shape_[1];

        if (r_rows != l_cols)
            throw std::invalid_argument("Invalid shapes for matrix multiplication.");

        Tensor<T> res(l_rows, r_cols);
        for (int r = 0; r < l_rows; r++)
        {
            for (int c = 0; c < r_cols; c++)
            {
                T tmp = 0;
                for (int k = 0; k < l_cols; k++)
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
        for (int i = 0; i < t.size_; i++)
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
    int size_;
    std::vector<int> shape_;

    std::shared_ptr<T[]> data_;

    Tensor(std::vector<int> shape, std::shared_ptr<T[]> data)
        : shape_(shape)
    {
        size_ = compute_size(shape);
        data_ = data;
    }

    int compute_size(std::vector<int> shape) const
    {
        int res = 1;
        for (auto v : shape)
            res *= v;
        return res;
    }

    int coord_to_index(const std::vector<int>& coords) const
    {
        assert(coords.size() == shape_.size());

        int res = 0;
        int step = 1;

        for (int dim = shape_.size() - 1; dim >= 0; dim--)
        {
            res += coords[dim] * step;
            step *= shape_[dim];
        }

        assert(res < size_);
        return res;
    }

    float get_random_float(void) const
    {
        static constexpr float min = -1.0f;
        static constexpr float max = 1.0f;

        float d = static_cast<float>(RAND_MAX / (max - min));

        return min + static_cast<float>(rand() / d);
    }
};
