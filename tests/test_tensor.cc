#include <iostream>
#include <criterion/criterion.h>
#include <vector>

#include "../src/tensor/tensor.hh"

static inline int add(int a, int b)
{
    return a + b;
}

Test(test_tensor, simple_test)
{
    Tensor<int> t({ 2, 2, 3, 3 });
    t.fill(fill_type::SEQUENCE);

    Tensor<int> t2({ 2, 2, 3, 3 });
    t2.fill(fill_type::ONES);

    t += t2;

    // std::cout << t;

    cr_assert_eq(t({ 0, 0, 0, 0 }), 1);
    cr_assert_eq(t({ 0, 0, 1, 1 }), 5);
    cr_assert_eq(t({ 0, 0, 2, 2 }), 9);
}

Test(test_tensor, test_reduce_one_dim)
{
    Tensor<int> t({ 3, 2 });
    t.fill(fill_type::SEQUENCE);

    Tensor<int> please = t.reduce(1, 0, add);

    cr_assert_eq(please({ 0 }), 1);
    cr_assert_eq(please({ 1 }), 5);
    cr_assert_eq(please({ 2 }), 9);
}

Test(test_tensor, test_reduce_all_axis)
{
    Tensor<int> t({ 3, 2, 2 });
    t.fill(fill_type::SEQUENCE);

    Tensor<int> please = t.reduce({ 0, 1, 2 }, 0, add);

    cr_assert_eq(please({ 0 }), 66);
}

Test(test_tensor, test_reduce_two_dim)
{
    Tensor<int> t({ 3, 2, 2 });
    t.fill(fill_type::SEQUENCE);

    Tensor<int> please = t.reduce(1, 0, add);

    cr_assert_eq(please({ 0, 0 }), 2);
    cr_assert_eq(please({ 0, 1 }), 4);
    cr_assert_eq(please({ 1, 0 }), 10);
    cr_assert_eq(please({ 1, 1 }), 12);
    cr_assert_eq(please({ 2, 0 }), 18);
    cr_assert_eq(please({ 2, 1 }), 20);
}

Test(test_tensor, test_reduce_three_dim)
{
    Tensor<int> t({ 2, 2, 2, 2 });
    t.fill(fill_type::SEQUENCE);

    Tensor<int> please = t.reduce(1, 0, add);

    cr_assert_eq(please({ 0, 0, 0 }), 4);
    cr_assert_eq(please({ 0, 0, 1 }), 6);
    cr_assert_eq(please({ 0, 1, 0 }), 8);
    cr_assert_eq(please({ 0, 1, 1 }), 10);
    cr_assert_eq(please({ 1, 0, 0 }), 20);
    cr_assert_eq(please({ 1, 0, 1 }), 22);
    cr_assert_eq(please({ 1, 1, 0 }), 24);
    cr_assert_eq(please({ 1, 1, 1 }), 26);
}

Test(test_tensor, test_reduce_two_axis)
{
    Tensor<int> t({ 3, 2, 2 });
    t.fill(fill_type::SEQUENCE);

    std::vector<int> axis{ 1, 2, 2 };
    Tensor<int> please = t.reduce(axis, 0, add);

    cr_assert_eq(please({ 0 }), 6);
    cr_assert_eq(please({ 1 }), 22);
    cr_assert_eq(please({ 2 }), 38);
}

Test(test_tensor, test_reduce_axis_zero)
{
    Tensor<int> t({ 3, 2, 2 });
    t.fill(fill_type::SEQUENCE);

    Tensor<int> please = t.reduce(0, 0, add);

    cr_assert_eq(please({ 0, 0 }), 12);
    cr_assert_eq(please({ 0, 1 }), 15);
    cr_assert_eq(please({ 1, 0 }), 18);
    cr_assert_eq(please({ 1, 1 }), 21);

}
