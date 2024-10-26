#ifndef VALUE_HPP
#define VALUE_HPP
#include <iostream>
#include <set>
#include <vector>
#include <cmath>
#include <functional>
#include <memory>

using namespace std;
class Value : public enable_shared_from_this<Value>
{
public:
    // Optional field, to help with debugging
    string l;

    // Constructor
    Value(float d, vector<shared_ptr<Value>> p = {}) : data{d}, grad{0}, prev{p}, _backward{[](shared_ptr<Value> &self) {}} {};

    // Getters and setters
    float getData();
    float getGrad();

    void setData(float d);
    void setGrad(float g);

    vector<shared_ptr<Value>> *get_prev()
    {
        return &prev;
    };

    // Friend methods
    friend void build_topo(const shared_ptr<Value> &v, vector<shared_ptr<Value>> &topo, set<shared_ptr<Value>> &visited);
    friend shared_ptr<Value> min(const shared_ptr<Value> &a, const shared_ptr<Value> &b);
    friend shared_ptr<Value> max(const shared_ptr<Value> &a, const shared_ptr<Value> &b);

    friend shared_ptr<Value> sum(vector<shared_ptr<Value>> &args);
    friend shared_ptr<Value> operator+(const shared_ptr<Value> &a, const shared_ptr<Value> &b);
    friend shared_ptr<Value> operator-(const shared_ptr<Value> &a, const shared_ptr<Value> &b);
    friend shared_ptr<Value> operator-(const shared_ptr<Value> &a);
    friend shared_ptr<Value> operator*(const shared_ptr<Value> &a, const shared_ptr<Value> &b);
    friend shared_ptr<Value> operator/(const shared_ptr<Value> &a, const shared_ptr<Value> &b);
    friend shared_ptr<Value> operator^(const shared_ptr<Value> &v, float p);
    friend shared_ptr<Value> tanh(shared_ptr<Value> v);
    friend shared_ptr<Value> relu(shared_ptr<Value> v);
    friend shared_ptr<Value> exp(const shared_ptr<Value> &v);
    friend shared_ptr<Value> log(const shared_ptr<Value> &v);
    friend ostream &operator<<(ostream &out, Value &v);

    // Functional
    void setBackward(function<void(shared_ptr<Value> &self)> funct);
    void backward();

private:
    float data;
    float grad;
    vector<shared_ptr<Value>> prev;
    function<void(shared_ptr<Value> &self)> _backward;
};

#endif