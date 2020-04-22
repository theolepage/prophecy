#include <iostream>
#include <criterion/criterion.h>

#include "../src/matrix/matrix.hh"

Test(test_matrix, simple_get)
{
    Matrix<int> m(2, 2);
    m.fill(fill_type::SEQUENCE);

    m(0, 1) = 17;

    // Matrix<int> m2(8, 8);
    // m2.fill(fill_type::SEQUENCE);
    // std::cout << m2;

    cr_assert_eq(m(0, 0), 0);
    cr_assert_eq(m(0, 1), 17);
    cr_assert_eq(m(1, 0), 2);
    cr_assert_eq(m(1, 1), 3);
}

Test(test_matrix, simple_transpose)
{
    Matrix<int> m(3, 3);
    m.fill(fill_type::SEQUENCE);

    Matrix<int> res = m.transpose();

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
    Matrix<int> m(2, 2);
    m.fill(fill_type::SEQUENCE);

    std::function<double(double)> f = [](double x) { return 2 * x; };
    m = m.map(f);

    cr_assert_eq(m(0, 0), 0);
    cr_assert_eq(m(0, 1), 2);
    cr_assert_eq(m(1, 0), 4);
    cr_assert_eq(m(1, 1), 6);
}

Test(test_matrix, simple_addition)
{
    Matrix<int> a(2, 2);
    Matrix<int> b(2, 2);
    a.fill(fill_type::SEQUENCE);
    b.fill(fill_type::SEQUENCE);

    // Matrix<int> res = a + b;

    // cr_assert_eq(res(0, 0), 0);
    // cr_assert_eq(res(0, 1), 2);
    // cr_assert_eq(res(1, 0), 4);
    // cr_assert_eq(res(1, 1), 6);
}

Test(test_matrix, simple_substraction)
{
    Matrix<int> a(2, 2);
    Matrix<int> b(2, 2);
    a.fill(fill_type::SEQUENCE);
    b.fill(fill_type::SEQUENCE);

    Matrix<int> res = a - b;

    cr_assert_eq(res(0, 0), 0);
    cr_assert_eq(res(0, 1), 0);
    cr_assert_eq(res(1, 0), 0);
    cr_assert_eq(res(1, 1), 0);
}

Test(test_matrix, simple_multiplication)
{
    Matrix<int> a(2, 2);
    Matrix<int> b(2, 2);
    a.fill(fill_type::SEQUENCE);
    b.fill(fill_type::SEQUENCE);

    // Matrix<int> res = a * b;

    // cr_assert_eq(res(0, 0), 2);
    // cr_assert_eq(res(0, 1), 3);
    // cr_assert_eq(res(1, 0), 6);
    // cr_assert_eq(res(1, 1), 11);
}

Test(test_matrix, simple_multiply)
{
    Matrix<int> a(2, 2);
    Matrix<int> b(2, 2);
    a.fill(fill_type::SEQUENCE);
    b.fill(fill_type::SEQUENCE);

    Matrix<int> res = Matrix<int>::multiply(a, b);

    cr_assert_eq(res(0, 0), 0);
    cr_assert_eq(res(0, 1), 1);
    cr_assert_eq(res(1, 0), 4);
    cr_assert_eq(res(1, 1), 9);
}
