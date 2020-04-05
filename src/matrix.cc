#include <ctime>
#include <cstdlib>
#include "matrix.hh"

Matrix::Matrix(unsigned rows, unsigned cols)
    : rows_(rows), cols_(cols), data_(rows * cols)
{}

Matrix::Matrix(const Matrix& m)
    : rows_(m.rows_), cols_(m.cols_), data_(m.data_)
{}

Matrix& Matrix::operator=(const Matrix& m)
{
    rows_ = m.rows_;
    cols_ = m.cols_;
    data_ = m.data_;
    return *this;
}

unsigned Matrix::get_rows() const
{
    return rows_;
}

unsigned Matrix::get_cols() const
{
    return cols_;
}

double& Matrix::operator()(unsigned row, unsigned col)
{
    return data_.at(row * cols_ + col);
}

std::ostream& operator<<(std::ostream& os, Matrix& m)
{
    for (unsigned r = 0; r < m.get_rows(); r++)
    {
        os << m(r, 0);
        for (unsigned c = 1; c < m.get_cols(); c++)
            os << "\t" << m(r, c);
        os << std::endl;
    }
    return os;
}

Matrix Matrix::transpose()
{
    Matrix res(cols_, rows_);
    for (unsigned i = 0; i < rows_; i++)
    {
        for (unsigned j = 0; j <= i; j++)
        {
            res.data_[i * cols_ + j] = data_[j * cols_ + i];
            res.data_[j * cols_ + i] = data_[i * cols_ + j];
        }
    }
    return res;
}

void Matrix::fill_random()
{
    srand (time(NULL));
    for (unsigned i = 0; i < rows_ * cols_; i++)
        data_[i] = (std::rand() / (double)RAND_MAX) * 2 - 1;
}

void Matrix::fill_sequence()
{
    for (unsigned i = 0; i < rows_ * cols_; i++)
        data_[i] = i;
}

Matrix& Matrix::map(std::function<double(double)> func)
{
    for (unsigned r = 0; r < rows_; r++)
        for (unsigned c = 0; c < cols_; c++)
            data_[r * cols_ + c] = func(data_[r * cols_ + c]);
    return *this;
}

Matrix Matrix::multiply(const Matrix& a, const Matrix& b)
{
    if (a.rows_ != b.rows_ || a.cols_ != b.cols_)
        throw "Invalid matrix shape";

    Matrix res(a);
    for (unsigned r = 0; r < a.rows_; r++)
        for (unsigned c = 0; c < a.cols_; c++)
            res.data_[r * a.cols_ + c] *= b.data_[r * a.cols_ + c];
    return res;
}

Matrix& Matrix::operator+=(const Matrix& m)
{
    if (m.rows_ != rows_ || m.cols_ != cols_)
        throw "Invalid matrix shape";

    for (unsigned r = 0; r < rows_; r++)
        for (unsigned c = 0; c < cols_; c++)
            data_[r * cols_ + c] += m.data_[r * cols_ + c];
    return *this;
}

Matrix operator+(const Matrix& a, const Matrix& b)
{
    Matrix res(a);
    return (res += b);
}

Matrix& Matrix::operator-=(const Matrix& m)
{
    if (m.rows_ != rows_ || m.cols_ != cols_)
        throw "Invalid matrix shape";

    for (unsigned r = 0; r < rows_; r++)
        for (unsigned c = 0; c < cols_; c++)
            data_[r * cols_ + c] -= m.data_[r * cols_ + c];
    return *this;
}

Matrix operator-(const Matrix& a, const Matrix& b)
{
    Matrix res(a);
    return (res -= b);
}

Matrix& Matrix::operator*=(const Matrix& m)
{
    if (m.rows_ != cols_)
        throw "Invalid matrix shape";

    Matrix res(rows_, m.cols_);
    for (unsigned r = 0; r < rows_; r++)
    {
        for (unsigned c = 0; c < m.cols_; c++)
        {
            double tmp = 0;
            for (unsigned k = 0; k < cols_; k++)
                tmp += data_[r * cols_ + k] * m.data_[k * m.cols_ + c];
            res.data_[r * m.cols_ + c] = tmp;
        }
    }
    *this = res;
    return *this;
}

Matrix operator*(const Matrix& a, const Matrix& b)
{
    Matrix res(a);
    return (res *= b);
}
