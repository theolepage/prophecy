#include <ctime>
#include <cstdlib>
#include "matrix.hh"

Matrix::Matrix(unsigned rows, unsigned cols)
    : rows_(rows), cols_(cols), data_(rows * cols)
{}

double& Matrix::operator[](long index)
{
    return data_.at(index);
}

double& Matrix::get(long row, long col)
{
    return data_.at(row * cols_ + col);
}

void Matrix::random()
{
    for (unsigned i = 0; i < rows_ * cols_; i++)
		data_[i] = (std::rand() / (double)RAND_MAX) * 2 - 1;
}
