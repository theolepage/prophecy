#include "gtest/gtest.h"

#include <iostream>
#include <vector>

#include "../src/tensor/tensor.hh"

static inline int add(int a, int b)
{
    return a + b;
}

TEST(test_matrix, simple_get)
{
    Tensor<int> m({ 2, 2 });
    m.fill(fill_type::SEQUENCE);

    m(0, 1) = 17;

    ASSERT_EQ(m(0, 0), 0);
    ASSERT_EQ(m(0, 1), 17);
    ASSERT_EQ(m(1, 0), 2);
    ASSERT_EQ(m(1, 1), 3);
}

TEST(test_matrix, simple_transpose)
{
    Tensor<int> m({ 3, 3 });
    m.fill(fill_type::SEQUENCE);

    Tensor<int> res = m.transpose();

    ASSERT_EQ(res(0, 0), 0);
    ASSERT_EQ(res(1, 0), 1);
    ASSERT_EQ(res(2, 0), 2);
    ASSERT_EQ(res(0, 1), 3);
    ASSERT_EQ(res(1, 1), 4);
    ASSERT_EQ(res(2, 1), 5);
    ASSERT_EQ(res(0, 2), 6);
    ASSERT_EQ(res(1, 2), 7);
    ASSERT_EQ(res(2, 2), 8);
}

TEST(test_matrix, simple_map)
{
    Tensor<int> m({ 2, 2 });
    m.fill(fill_type::SEQUENCE);

    std::function<double(double)> f = [](double x) { return 2 * x; };
    m = m.map(f);

    ASSERT_EQ(m(0, 0), 0);
    ASSERT_EQ(m(0, 1), 2);
    ASSERT_EQ(m(1, 0), 4);
    ASSERT_EQ(m(1, 1), 6);
}

TEST(test_matrix, simple_addition)
{
    Tensor<int> a({ 2, 2 });
    Tensor<int> b({ 2, 2 });
    a.fill(fill_type::SEQUENCE);
    b.fill(fill_type::SEQUENCE);

    Tensor<int> res = a + b;

    ASSERT_EQ(res(0, 0), 0);
    ASSERT_EQ(res(0, 1), 2);
    ASSERT_EQ(res(1, 0), 4);
    ASSERT_EQ(res(1, 1), 6);
}

TEST(test_matrix, simple_substraction)
{
    Tensor<int> a({ 2, 2 });
    Tensor<int> b({ 2, 2 });
    a.fill(fill_type::SEQUENCE);
    b.fill(fill_type::SEQUENCE);

    Tensor<int> res = a - b;

    ASSERT_EQ(res(0, 0), 0);
    ASSERT_EQ(res(0, 1), 0);
    ASSERT_EQ(res(1, 0), 0);
    ASSERT_EQ(res(1, 1), 0);
}

TEST(test_matrix, simple_multiplication)
{
    Tensor<int> a({ 2, 2 });
    Tensor<int> b({ 2, 2 });
    a.fill(fill_type::SEQUENCE);
    b.fill(fill_type::SEQUENCE);

    Tensor<int> res = a.matmul(b);

    ASSERT_EQ(res(0, 0), 2);
    ASSERT_EQ(res(0, 1), 3);
    ASSERT_EQ(res(1, 0), 6);
    ASSERT_EQ(res(1, 1), 11);
}

TEST(test_matrix, simple_multiply)
{
    Tensor<int> a({ 2, 2 });
    Tensor<int> b({ 2, 2 });
    a.fill(fill_type::SEQUENCE);
    b.fill(fill_type::SEQUENCE);

    Tensor<int> res = a * b;

    ASSERT_EQ(res(0, 0), 0);
    ASSERT_EQ(res(0, 1), 1);
    ASSERT_EQ(res(1, 0), 4);
    ASSERT_EQ(res(1, 1), 9);
}

TEST(test_tensor, simple_test)
{
    Tensor<int> t({ 2, 2, 3, 3 });
    t.fill(fill_type::SEQUENCE);

    Tensor<int> t2({ 2, 2, 3, 3 });
    t2.fill(fill_type::ONES);

    t += t2;

    // std::cout << t;

    ASSERT_EQ(t({ 0, 0, 0, 0 }), 1);
    ASSERT_EQ(t({ 0, 0, 1, 1 }), 5);
    ASSERT_EQ(t({ 0, 0, 2, 2 }), 9);
}

TEST(test_tensor, test_reduce_one_dim)
{
    Tensor<int> t({ 3, 2 });
    t.fill(fill_type::SEQUENCE);

    Tensor<int> please = t.reduce(1, 0, add);

    ASSERT_EQ(please({ 0 }), 1);
    ASSERT_EQ(please({ 1 }), 5);
    ASSERT_EQ(please({ 2 }), 9);
}

TEST(test_tensor, test_reduce_all_axis)
{
    Tensor<int> t({ 3, 2, 2 });
    t.fill(fill_type::SEQUENCE);

    Tensor<int> please = t.reduce({ 0, 1, 2 }, 0, add);

    ASSERT_EQ(please({ 0 }), 66);
}

TEST(test_tensor, test_reduce_two_dim)
{
    Tensor<int> t({ 3, 2, 2 });
    t.fill(fill_type::SEQUENCE);

    Tensor<int> please = t.reduce(1, 0, add);

    ASSERT_EQ(please({ 0, 0 }), 2);
    ASSERT_EQ(please({ 0, 1 }), 4);
    ASSERT_EQ(please({ 1, 0 }), 10);
    ASSERT_EQ(please({ 1, 1 }), 12);
    ASSERT_EQ(please({ 2, 0 }), 18);
    ASSERT_EQ(please({ 2, 1 }), 20);
}

TEST(test_tensor, test_reduce_three_dim)
{
    Tensor<int> t({ 2, 2, 2, 2 });
    t.fill(fill_type::SEQUENCE);

    Tensor<int> please = t.reduce(1, 0, add);

    ASSERT_EQ(please({ 0, 0, 0 }), 4);
    ASSERT_EQ(please({ 0, 0, 1 }), 6);
    ASSERT_EQ(please({ 0, 1, 0 }), 8);
    ASSERT_EQ(please({ 0, 1, 1 }), 10);
    ASSERT_EQ(please({ 1, 0, 0 }), 20);
    ASSERT_EQ(please({ 1, 0, 1 }), 22);
    ASSERT_EQ(please({ 1, 1, 0 }), 24);
    ASSERT_EQ(please({ 1, 1, 1 }), 26);
}

TEST(test_tensor, test_reduce_two_axis)
{
    Tensor<int> t({ 3, 2, 2 });
    t.fill(fill_type::SEQUENCE);

    std::vector<int> axis{ 1, 2, 2 };
    Tensor<int> please = t.reduce(axis, 0, add);

    ASSERT_EQ(please({ 0 }), 6);
    ASSERT_EQ(please({ 1 }), 22);
    ASSERT_EQ(please({ 2 }), 38);
}

TEST(test_tensor, test_reduce_axis_zero)
{
    Tensor<int> t({ 3, 2, 2 });
    t.fill(fill_type::SEQUENCE);

    Tensor<int> please = t.reduce(0, 0, add);

    ASSERT_EQ(please({ 0, 0 }), 12);
    ASSERT_EQ(please({ 0, 1 }), 15);
    ASSERT_EQ(please({ 1, 0 }), 18);
    ASSERT_EQ(please({ 1, 1 }), 21);
}

TEST(test_tensor, get_one)
{
    Tensor<int> t({ 3, 2, 2 });
    t.fill(fill_type::SEQUENCE);

    Tensor<int> please = t.extract({ 1, 1 });

    ASSERT_EQ(please({ 0 }), 6);
    ASSERT_EQ(please({ 1 }), 7);
}

TEST(test_tensor, simple_get)
{
    Tensor<int> t({ 2, 2, 3, 3 });
    t.fill(fill_type::SEQUENCE);

    Tensor<int> please = t.extract({ 1, 1, 2 });

    ASSERT_EQ(please({ 0 }), 33);
    ASSERT_EQ(please({ 1 }), 34);
    ASSERT_EQ(please({ 2 }), 35);
}

int main(int argc, char *argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}