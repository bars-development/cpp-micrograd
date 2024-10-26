#include "../include/NN.hpp"
#include <iostream>
#include <cassert>
#include <sstream>
#include <cmath>

// Helper function to compare floating point numbers
bool is_close(double a, double b, double tol = 1e-6)
{
    return std::fabs(a - b) < tol;
}

void test_neuron_forward_complex()
{
    // Setup test case with specific weights and bias for Neuron
    Neuron n({float(int(activation::tanh)),
              0.4, -0.6, 0.8, 0.1});

    vector<shared_ptr<Value>> inputs = {
        make_shared<Value>(1.0),
        make_shared<Value>(-2.0),
        make_shared<Value>(0.5)};

    auto output = n(inputs);

    // Expected output manually computed
    double z = 0.4 * 1.0 + (-0.6) * (-2.0) + 0.8 * 0.5 + 0.1; // z = 1.8
    double expected = tanh(z);

    assert(is_close(output->getData(), expected));
    cout << "Neuron forward pass (complex) test passed." << endl;
}

void test_linear_layer_forward_complex()
{
    // Setup test case with specific weights for LinearLayer

    istringstream s("3\
    2\
        2 0.3 -0.5 0.7 0.2\
        2 0.1 0.4 -0.3 -0.1");
    LinearLayer layer(s);

    vector<shared_ptr<Value>> inputs = {
        make_shared<Value>(0.5),
        make_shared<Value>(1.2),
        make_shared<Value>(-0.7)};

    auto output = layer(inputs);

    // Expected output manually computed
    double z1 = 0.3 * 0.5 + (-0.5) * 1.2 + 0.7 * (-0.7) + 0.2; // z1 = -0.54
    double z2 = 0.1 * 0.5 + 0.4 * 1.2 + (-0.3) * (-0.7) - 0.1; // z2 = 0.82

    // cout << output[0];
    assert(output.size() == 2);
    assert(is_close(output[0]->getData(), std::max(0.0, z1))); // ReLU(z1)
    assert(is_close(output[1]->getData(), std::max(0.0, z2))); // ReLU(z2)

    cout << "Linear layer forward pass (complex) test passed." << endl;
}

void test_mlp_forward_complex()
{
    // Setup test case with specific weights and architecture
    MLP mlp(3, {3, 2});
    // auto &layers = mlp;

    vector<float> inputs = {0.3f, -0.5f, 0.8f};
    vector<shared_ptr<Value>> inp;
    for (auto i : inputs)
    {
        inp.push_back(make_shared<Value>(i));
    }
    auto output = mlp(inp);

    assert(output.size() == 2);
    cout << "MLP forward pass (complex) test passed." << endl;
}

int main()
{
    test_neuron_forward_complex();
    test_linear_layer_forward_complex();
    test_mlp_forward_complex();
    cout << "All NN detailed tests passed!" << endl;
    return 0;
}
