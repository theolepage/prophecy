#pragma once

#include <ostream>
#include <memory>
#include <functional>
#include <cassert>
#include <cstdarg>

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
        : size_(0), shape_(0), data_(nullptr)
    {}

    Tensor(std::initializer_list<int> shape)
        : shape_(shape)
    {
        size_ = compute_size(shape);
        data_ = std::shared_ptr<T[]>(new T[size_]);
    }

    Tensor(std::vector<int> shape)
        : shape_(shape)
    {
        size_ = compute_size(shape);
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

    virtual ~Tensor() = default;

    /**
    * Getters / setters
    */

    std::vector<int> get_shape() const
    {
        return shape_;
    }

    int get_size() const
    {
        return size_;
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
        // To-Do: assert compute_size(shape) == compute_size(shape_)

        size_ = compute_size(shape);
        shape_ = shape;
    }

    /**
    * Operations (fill, map, op)
    */

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

    Tensor<T> op(const Tensor &right, std::function<T(T, T)> fn)
    {
        assert(data_ != nullptr);
        if (shape_ != right.shape_)
            throw std::invalid_argument("Operations requires the two tensors to have the same shape.");

        Tensor<T> res(shape_);
        for (int i = 0; i < get_size(); i++)
            res.data_[i] = fn(data_[i], right.data_[i]);
        return res;
    }

    Tensor<T>& op_inplace(const Tensor &right, std::function<T(T, T)> fn)
    {
        assert(data_ != nullptr);
        if (shape_ != right.shape_)
            throw std::invalid_argument("Operations requires the two tensors to have the same shape.");

        for (int i = 0; i < get_size(); i++)
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

    Tensor<T> sum(std::vector<int> axis)
    {
        Tensor<T> res = *this;
        for (int dim : axis)
        {
            res = res.sum(dim);
        }
        return res;
    }

    Tensor<T> sum(unsigned dim)
    {
        // Compute output shape
        std::vector<int> output_shape;
        for (unsigned i = 0; i < shape_.size(); i++)
            if (i != dim)
                output_shape.push_back(shape_[i]);
        Tensor<T> res(output_shape);
        int index = 0;

        // Compute the step
        int step = 1;
        for (unsigned i = dim + 1; i < shape_.size(); i++)
            step *= shape_[i];

        // For every element of this dimension, sum what is inside
        int size_prevdim = 1;
        for (unsigned i = 0; i < dim; i++)
            size_prevdim *= shape_[i];
        for (int i = 0; i < size_prevdim; i++)
            sum(res, i, step, dim, index);

        return res;
    }

    void sum(const Tensor<T>& res, unsigned i, int step, int dim, int& index)
    {
        int size_dim = shape_[dim];
        unsigned begin = i * (step * size_dim);
        for (int i = 0; i < step; i++)
        {
            T sum = 0;
            for (int j = 0; j < size_dim; j++)
                sum += data_[begin + j * step + i];
            res.data_[index] = sum;
            index += 1;
        }
    }

    Tensor<T> conv(const Tensor &kernel, int padding, int stride)
    {
        //           H  W  C
        // kernel: ( 5, 5, 3 )

        // for each filter input.conv(filter, padding, stride)
        // stack 
    }

    /**
    * Matrix operations (transpose, matmul)
    */

    Tensor<T> transpose()
    {
        assert(data_ != nullptr && shape_.size() == 2);

        Tensor<T> res(shape_);
        int rows = shape_[0];
        int cols = shape_[1];

        for (unsigned i = 0; i < rows; i++)
            for (unsigned j = 0; j < cols; j++)
                res.data_[j * rows + i] = data_[i * cols + j];

        return res;
    }

    Tensor<T> matmul(const Tensor<T>& right)
    {
        assert(data_ != nullptr && shape_.size() == 2);

        int l_rows = shape_[0];
        int l_cols = shape_[1];
        int r_rows = right.shape_[0];
        int r_cols = right.shape_[1];

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
        for (int i = 0; i < t.get_size(); i++)
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

    int compute_size(std::vector<int> shape)
    {
        int res = 1;
        for (auto v : shape)
            res *= v;
        return res;
    }

    int coord_to_index(std::vector<int> coords)
    {
        assert(coords.size() <= shape_.size());

        int res = 0;
        for (unsigned i = 0; i < coords.size(); i++)
        {
            unsigned indices_to_skip = 1;
            for (unsigned j = i + 1; j < shape_.size(); j++)
                indices_to_skip *= shape_[j];
            res += coords[i] * indices_to_skip;   
        }

        assert(res < size_);
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
