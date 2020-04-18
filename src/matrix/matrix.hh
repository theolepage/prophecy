#pragma once

#include <ostream>
#include <memory>
#include <functional>
#include <cassert>

enum class transpose
{
    LEFT,
    RIGHT,
    NO_IMPLICIT
};

enum class fill_type
{
    RANDOM_FLOAT,
    SEQUENCE,
    ZERO
};

template <typename T>
class Matrix
{
public:
    Matrix()
    {
        data_ = nullptr;
        rows_ = 0;
        cols_ = 0;
    }

    Matrix(int rows, int cols) : rows_(rows), cols_(cols)
    {
        data_ = std::shared_ptr<T[]>(new T[cols_ * rows_]);
    }

    Matrix(const Matrix<T>& m)
    {
        data_ = m.data_;
        rows_ = m.rows_;
        cols_ = m.cols_;
    }

    Matrix& operator=(const Matrix<T>& m)
    {
        data_ = m.data_;
        rows_ = m.rows_;
        cols_ = m.cols_;
        return *this;
    }

    // Lazy constructor for performance reason
    // TODO useful ?
    Matrix(T* array, int height, int width) : rows_(height), cols_(width)
    {
        data_ = std::shared_ptr<T[]>(array);
    }

    virtual ~Matrix() = default;

    void fill(std::function<T(void)> value_initializer)
    {
        assert(data_ != nullptr);
        for (int i = 0; i < rows_ * cols_; ++i)
            data_[i] = value_initializer();
    }

    void fill(T value)
    {
        assert(data_ != nullptr);
        for (int i = 0; i < rows_ * cols_; ++i)
            data_[i] = value;
    }

    void fill(fill_type type)
    {
        assert(data_ != nullptr);
        switch(type)
        {
            case fill_type::RANDOM_FLOAT:
                for (int i = 0; i < rows_ * cols_; ++i)
                    data_[i] = get_random_float();
                break;
            case fill_type::SEQUENCE:
                for (int i = 0; i < rows_ * cols_; ++i)
                    data_[i] = i;
                break;
            case fill_type::ZERO:
                for (int i = 0; i < rows_ * cols_; ++i)
                    data_[i] = static_cast<T>(0);
                break;
        }
    }

    Matrix<T>& map_inplace(std::function<T(T)> value_initializer)
    {
        assert(data_ != nullptr);
        for (int i = 0; i < rows_ * cols_; ++i)
            data_[i] = value_initializer(data_[i]);
        return *this;
    }

    Matrix<T> map(std::function<T(T)> value_initializer) const
    {
        assert(data_ != nullptr);
        Matrix<T> result(rows_, cols_);
        for (int i = 0; i < rows_ * cols_; ++i)
            result.data_[i] = value_initializer(data_[i]);
        return result;
    }

    T operator()(int y, int x) const
    {
        assert(y * cols_ + x < cols_ * rows_);
        return data_[y * cols_ + x];
    }

    T& operator()(int y, int x)
    {
        assert(y * cols_ + x < cols_ * rows_);
        return data_[y * cols_ + x];
    }

    bool operator==(const Matrix &right)
    {
        if (cols_ != right.cols_ || rows_ != right.rows_)
            throw std::invalid_argument("+ on Matrix need same width and height");
        for (int i = 0; i < rows_; ++i)
        {
            for (int j = 0; j < cols_; ++j)
                if ((*this)(i, j) != right(i, j))
                    return false;
        }
        return true;
    }

    Matrix<T>& multiply_inplace(const Matrix& b)
    {
        if (rows_ != b.rows_ || cols_ != b.cols_)
            throw "Invalid matrix shape";

        for (int i = 0; i < rows_ * cols_; i++)
            data_[i] *= b.data_[i];

        return *this;
    }

    static Matrix<T> multiply(const Matrix& a, const Matrix& b)
    {
        if (a.rows_ != b.rows_ || a.cols_ != b.cols_)
            throw "Invalid matrix shape";

        Matrix res(a);
        for (int i = 0; i < a.rows_ * a.cols_; i++)
            res.data_[i] *= b.data_[i];
        return res;
    }

    static Matrix<T> dot(const Matrix& left, const Matrix& right)
    {
        return dot(left, right, transpose::NO_IMPLICIT);
    }

    static Matrix<T> dot(const Matrix& left, const Matrix &right, transpose order)
    {
        if (left.cols_ == right.rows_)
        {
            Matrix result(left.rows_, right.cols_);
            float sum = 0.0f;
            for (int y = 0; y < left.rows_; ++y)
            {
                for (int x = 0; x < right.cols_; ++x)
                {
                    for (int k = 0; k < left.cols_; k++)
                        sum += left(y, k) * right(k, x);
                    result(y, x) = sum;
                    sum = 0;
                }
            }
            return result;
        }
        else if (left.rows_ == right.rows_ && order == transpose::LEFT) // Implicit left transpose
        {
            Matrix result(left.cols_, right.cols_);
            float sum = 0.0f;
            for (int x = 0; x < left.cols_; ++x)
            {
                for (int y = 0; y < right.cols_; ++y)
                {
                    for (int k = 0; k < left.rows_; k++)
                        sum += left(k, x) * right(k, y);
                    result(x, y) = sum;
                    sum = 0;
                }
            }
            return result;
        }
        else if (left.cols_ == right.cols_ && order == transpose::RIGHT) // Implicit right transpose
        {
            Matrix result(left.rows_, right.rows_);
            float sum = 0.0f;
            for (int y = 0; y < left.rows_; ++y)
            {
                for (int x = 0; x < right.rows_; ++x)
                {
                    for (int k = 0; k < right.cols_; k++)
                        sum += left(y, k) * right(x, k);
                    result(y, x) = sum;
                    sum = 0;
                }
            }
            return result;
        }
        else
            throw std::invalid_argument("Bad Matrix multiplication");
    }

    Matrix<T> transpose(void) const
    {
        Matrix trans(cols_, rows_);
        for (int y = 0; y < rows_; ++y)
            for (int x = 0; x < cols_; ++x)
                trans(x, y) = (*this)(y, x);
        return trans;
    }

    Matrix<T>& operator+=(const Matrix &right)
    {
        if (cols_ != right.cols_ || rows_ != right.rows_)
            throw std::invalid_argument("+ on Matrix need same width and height");
        for (int i = 0; i < rows_; ++i)
        {
            for (int j = 0; j < cols_; ++j)
                (*this)(i, j) += right(i, j);
        }
        return *this;
    }

    Matrix<T>& operator-=(const Matrix& right)
    {
        if (cols_ != right.cols_ || rows_ != right.rows_)
            throw std::invalid_argument("+ on Matrix need same width and height");

        for (int i = 0; i < rows_; ++i)
        {
            for (int j = 0; j < cols_; ++j)
                (*this)(i, j) -= right(i, j);
        }
        return *this;
    }

    Matrix<T> operator-(const Matrix& right)
    {
        if (cols_ != right.cols_ || rows_ != right.rows_)
            throw std::invalid_argument("+ on Matrix need same width and height");

        Matrix result(rows_, cols_);
        for (int i = 0; i < rows_; ++i)
        {
            for (int j = 0; j < cols_; ++j)
            {
                result(i, j) = (*this)(i, j);
                result(i, j) -= right(i, j);
            }
        }
        return result;
    }

    int get_cols(void) const { return cols_; }
    int get_rows(void) const { return rows_; }

    friend std::ostream& operator<<(std::ostream& os, Matrix& m)
    {
        for (int r = 0; r < m.get_rows(); r++)
        {
            os << m(r, 0);
            for (int c = 1; c < m.get_cols(); c++)
                os << "\t" << m(r, c);
            os << std::endl;
        }
        return os;
    }

private:
    int rows_;
    int cols_;
    std::shared_ptr<T[]> data_;

    float get_random_float(void)
    {
        static constexpr float min_rand_value = -1.0f;
        static constexpr float max_rand_value = 1.0f;
        return min_rand_value
                + static_cast <float>(rand()) /
                    (static_cast<float>(RAND_MAX/(max_rand_value-min_rand_value)));
    }
};
