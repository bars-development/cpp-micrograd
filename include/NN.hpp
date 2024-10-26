#ifndef NN_HPP
#define NN_HPP

#include <vector>
#include <memory>
#include <random>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include "ValueStruct.hpp"

using namespace std;

enum class activation
{
    none,
    tanh,
    relu
};

class Module
{
public:
    void zero_grad();
    virtual std::vector<std::shared_ptr<Value>> parameters() = 0;
};

// Class Neuron
class Neuron : public Module
{
public:
    Neuron(int nin, activation act);
    Neuron(vector<float> params);
    vector<shared_ptr<Value>> parameters();
    shared_ptr<Value> operator()(vector<shared_ptr<Value>> x);
    void save(ostream &out);

private:
    vector<shared_ptr<Value>> w;
    shared_ptr<Value> b;
    activation act;
};

// Class LinearLayer
class LinearLayer : public Module
{
public:
    LinearLayer(int nin, int nout, activation act);
    LinearLayer(istream &in);
    vector<shared_ptr<Value>> operator()(vector<shared_ptr<Value>> x);
    vector<shared_ptr<Value>> parameters();
    void save(ostream &out);

private:
    vector<Neuron> neurons;
    int nin;
    int nout;
};

// Class MLP
class MLP : public Module
{
public:
    MLP(int in, vector<int> l);
    MLP(string path);
    void saveTo(string path);
    vector<shared_ptr<Value>> operator()(vector<std::shared_ptr<Value>> input);
    vector<shared_ptr<Value>> parameters();

private:
    vector<LinearLayer> layers;
    vector<int> size;
};

// Function declarations
vector<shared_ptr<Value>> softMax(vector<shared_ptr<Value>> x);
shared_ptr<Value> simpleLoss(vector<shared_ptr<Value>> pred, vector<shared_ptr<Value>> y);
#endif
