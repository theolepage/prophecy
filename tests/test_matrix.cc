#include <iostream>
#include <criterion/criterion.h>

#include "../src/matrix.hh"

Test(test_matrix, simple_get)
{
    Matrix m(2, 2);
    m.fill_sequence();

    // Matrix m2(8, 8);
    // m2.fill_sequence();
    // std::cout << m2;

    cr_assert_eq(m(0, 0), 0);
    cr_assert_eq(m(0, 1), 1);
    cr_assert_eq(m(1, 0), 2);
    cr_assert_eq(m(1, 1), 3);
}

Test(test_matrix, simple_transpose)
{
    Matrix m(3, 3);
    m.fill_sequence();

    Matrix res = m.transpose();

    cr_assert_eq(res(0, 0), 0);
    cr_assert_eq(res(1, 0), 1);
    cr_assert_eq(res(2, 0), 2);
    cr_assert_eq(res(0, 1), 3);
    cr_assert_eq(res(1, 1), 4);
    cr_assert_eq(res(2, 1), 5);
    cr_assert_eq(res(0, 2), 6);
    cr_assert_eq(res(1, 2), 7);
    cr_assert_eq(res(2, 2), 8);
}

Test(test_matrix, simple_map)
{
    Matrix m(2, 2);
    m.fill_sequence();

    std::function<double(double)> f = [](double x) { return 2 * x; };
    m.map(f);

    cr_assert_eq(m(0, 0), 0);
    cr_assert_eq(m(0, 1), 2);
    cr_assert_eq(m(1, 0), 4);
    cr_assert_eq(m(1, 1), 6);
}

Test(test_matrix, simple_addition)
{
    Matrix a(2, 2);
    Matrix b(2, 2);
    a.fill_sequence();
    b.fill_sequence();

    Matrix res = a + b;

    cr_assert_eq(res(0, 0), 0);
    cr_assert_eq(res(0, 1), 2);
    cr_assert_eq(res(1, 0), 4);
    cr_assert_eq(res(1, 1), 6);
}

Test(test_matrix, simple_substraction)
{
    Matrix a(2, 2);
    Matrix b(2, 2);
    a.fill_sequence();
    b.fill_sequence();

    Matrix res = a - b;

    cr_assert_eq(res(0, 0), 0);
    cr_assert_eq(res(0, 1), 0);
    cr_assert_eq(res(1, 0), 0);
    cr_assert_eq(res(1, 1), 0);
}

Test(test_matrix, simple_multiplication)
{
    Matrix a(2, 2);
    Matrix b(2, 2);
    a.fill_sequence();
    b.fill_sequence();

    Matrix res = a * b;

    cr_assert_eq(res(0, 0), 2);
    cr_assert_eq(res(0, 1), 3);
    cr_assert_eq(res(1, 0), 6);
    cr_assert_eq(res(1, 1), 11);
}

Test(test_matrix, simple_multiply)
{
    Matrix a(2, 2);
    Matrix b(2, 2);
    a.fill_sequence();
    b.fill_sequence();

    Matrix res = Matrix::multiply(a, b);

    cr_assert_eq(res(0, 0), 0);
    cr_assert_eq(res(0, 1), 1);
    cr_assert_eq(res(1, 0), 4);
    cr_assert_eq(res(1, 1), 9);
}
