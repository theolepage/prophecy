#include <iostream>
#include <criterion/criterion.h>

#include "../src/tensor/tensor.hh"

Test(test_tensor, simple_test)
{
    Tensor<int> t({ 2, 2, 3, 3 });
    t.fill(fill_type::SEQUENCE);

    Tensor<int> t2({ 2, 2, 3, 3 });
    t2.fill(fill_type::ONES);

    t += t2;

    std::cout << t;

    cr_assert_eq(t({ 0, 0, 0, 0 }), 1);
    cr_assert_eq(t({ 0, 0, 1, 1 }), 5);
    cr_assert_eq(t({ 0, 0, 2, 2 }), 9);
}