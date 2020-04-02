#pragma once

#include <vector>

class Matrix
{
public:
    Matrix(unsigned rows, unsigned cols);
    double& operator()(long index);
    double& get(long row, long col);

    void random();

private:
    unsigned rows_;
    unsigned cols_;
    std::vector<double> data_;
};
