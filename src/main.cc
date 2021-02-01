#include <iostream>
#include <memory>

// #include "kernel.cuh"
// #include "layer/dense_layer.hh"
// #include "model/model.hh"
// #include "tensor/tensor.hh"

#include "xtensor-blas/xlinalg.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"

#include "pybind11/pybind11.h"
#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"

int test(xt::pyarray<double>& m)
{
    std::cout << "hello" << std::endl;
    std::cout << m << std::endl;
    return 17;
}

PYBIND11_MODULE(prophecy, m)
{
    xt::import_numpy();

    m.doc() = "Prophecy neural networks framework";

    m.def("test", test, "test");
}