#include "../include/ValueStruct.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

// Helper function to compare floating point numbers
bool is_close(double a, double b, double tol = 1e-6)
{
    return std::fabs(a - b) < tol;
}

void test_value_addition_complex()
{
    auto a = make_shared<Value>(5.0);
    auto b = make_shared<Value>(-3.2);
    auto c = make_shared<Value>(1.8);

    // Perform multiple additions
    auto result = a + b + c;

    // Expected result
    double expected = 5.0 + (-3.2) + 1.8; // result = 3.6

    assert(is_close(result->getData(), expected));
    cout << "Value addition (complex) test passed." << endl;
}

void test_value_multiplication_complex()
{
    auto x = make_shared<Value>(2.5);
    auto y = make_shared<Value>(-1.2);
    auto z = make_shared<Value>(3.0);

    // Perform multiple multiplications
    auto result = (x * y) * z;

    // Expected result
    double expected = 2.5 * (-1.2) * 3.0; // result = -9.0

    assert(is_close(result->getData(), expected));
    cout << "Value multiplication (complex) test passed." << endl;
}

void test_value_backward_complex()
{
    // Setup chain of operations
    auto x = make_shared<Value>(1.5);
    auto y = make_shared<Value>(-2.0);
    auto z = x + y;      // z = x + y
    auto w = z * z;      // w = z^2
    auto output = w + w; // output = 2 * z^2

    // Backpropagation
    output->backward();

    // Expected gradients:
    // dw/dz = 2 * z -> dz/dx = 1, dz/dy = 1, dw/dx = 4 * z, dw/dy = 4 * z
    assert(is_close(x->getGrad(), -2));
    assert(is_close(y->getGrad(), -2));

    cout << "Value backward (complex) test passed." << endl;
}

void test_value_chain_rule()
{
    // Test a more complex chain rule scenario
    auto a = make_shared<Value>(1.5);
    auto b = make_shared<Value>(2.0);
    auto c = make_shared<Value>(-1.0);

    auto d = a * b; // d = a * b
    auto e = d + c; // e = d + c
    auto f = e * e; // f = e^2

    f->backward();

    // df/de = 2 * e, de/dd = 1, dd/da = b, dd/db = a, de/dc = 1
    assert(is_close(a->getGrad(), 2 * (d + c)->getData() * b->getData()));
    assert(is_close(b->getGrad(), 2 * (d + c)->getData() * a->getData()));
    assert(is_close(c->getGrad(), 2 * (d + c)->getData()));

    cout << "Value chain rule test passed." << endl;
}

int main()
{
    test_value_addition_complex();
    test_value_multiplication_complex();
    test_value_backward_complex();
    test_value_chain_rule();
    cout << "All ValueStructure detailed tests passed!" << endl;
    return 0;
}
