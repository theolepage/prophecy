#pragma once

#include <ostream>
#include <vector>

class Matrix
{
public:
    Matrix(unsigned rows, unsigned cols);
    Matrix(const Matrix& m);
    Matrix& operator=(const Matrix& m);

    unsigned get_rows() const;
    unsigned get_cols() const;

    double& operator()(unsigned row, unsigned col);

    Matrix transpose();

    void fill_random();
    void fill_sequence();

    Matrix& operator+=(const Matrix& m);
    Matrix& operator-=(const Matrix& m);
    Matrix& operator*=(const Matrix& m);
    static Matrix multiply(const Matrix& a, const Matrix& b);

private:
    unsigned rows_;
    unsigned cols_;
    std::vector<double> data_;
};

std::ostream& operator<<(std::ostream& os, Matrix& m);
Matrix operator+(const Matrix& a, const Matrix& b);
Matrix operator-(const Matrix& a, const Matrix& b);
Matrix operator*(const Matrix& a, const Matrix& b);
