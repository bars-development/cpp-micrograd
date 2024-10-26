
#include "include/ValueStruct.hpp"
using namespace std;

float Value::getData()
{
    return data;
}
float Value::getGrad()
{
    return grad;
}

void Value::setData(float d)
{
    data = d;
}
void Value::setGrad(float g)
{
    grad = g;
}

void Value::setBackward(function<void(shared_ptr<Value> &self)> funct)
{
    _backward = funct;
}

shared_ptr<Value> min(const shared_ptr<Value> &a, const shared_ptr<Value> &b)
{
    if (a->data < b->data)
    {
        return a;
    }
    return b;
}
shared_ptr<Value> max(const shared_ptr<Value> &a, const shared_ptr<Value> &b)
{
    if (a->data > b->data)
    {
        return a;
    }
    return b;
}

shared_ptr<Value> sum(vector<shared_ptr<Value>> &args)
{
    float d = 0;
    for (auto &v : args)
    {
        d += v->getData();
    }
    shared_ptr<Value> out = make_shared<Value>(d, args);
    out->setBackward([](shared_ptr<Value> &self)
                     {
        for(auto &a: self->prev){
            a->grad += self->grad;
        } });

    return out;
}
shared_ptr<Value> operator+(const shared_ptr<Value> &a, const shared_ptr<Value> &b)
{
    float d = a->data + b->data;
    shared_ptr<Value> out = make_shared<Value>(d, vector{a, b});
    out->l = a->l + "+" + b->l;
    out->setBackward([](shared_ptr<Value> &self)
                     {
                         for(auto &a: self->prev){
            a->grad += 1 * self->grad;
        } });
    return out;
}
shared_ptr<Value> operator-(const shared_ptr<Value> &a)
{
    auto v = make_shared<Value>(-1);
    return a * v;
};
shared_ptr<Value> operator-(const shared_ptr<Value> &a, const shared_ptr<Value> &b)
{
    return a + (-b);
};
shared_ptr<Value> operator*(const shared_ptr<Value> &a, const shared_ptr<Value> &b)
{
    float d = a->data * b->data;
    shared_ptr<Value> out = make_shared<Value>(d, vector{a, b});
    out->l = a->l + "*" + b->l;

    out->setBackward([](shared_ptr<Value> &self)
                     {
                        auto it = (self->prev).begin();
                        if((self->prev).size()==1){
                            shared_ptr<Value> v1 = *it;
                            v1->grad += 2 * v1->data * self->grad;
                            return;
                        }
                        
                        shared_ptr<Value> v1 = *it;

                        shared_ptr<Value> v2 = *(++it);

                        v1->grad += v2->data * self->grad;
                        v2->grad += v1->data * self->grad; });
    return out;
}
shared_ptr<Value> operator/(const shared_ptr<Value> &a, const shared_ptr<Value> &b)
{
    auto out = a * (b ^ (-1));
    out->l = a->l + "/" + b->l;
    return out;
}
shared_ptr<Value> operator^(const shared_ptr<Value> &v, float p)
{
    float d = pow(v->data, p);
    shared_ptr<Value> out = make_shared<Value>(d, vector{v});
    out->setBackward([p](shared_ptr<Value> &self)
                     { 
                        
                        auto a = *(self->prev).begin();

                        a->grad = (p * pow(a->data, p - 1)) * self->grad; });
    out->l = v->l + "^" + to_string(p);
    return out;
}
shared_ptr<Value> exp(const shared_ptr<Value> &a)
{
    float d = exp(a->data);
    auto out = make_shared<Value>(d, vector{a});

    out->setBackward([d](shared_ptr<Value> &self)
                     { 
                        auto a = *(self->prev).begin();
                        a->grad += d * self->grad; });
    out->l = "exp(" + a->l + ")";
    return out;
}
shared_ptr<Value> log(const shared_ptr<Value> &v)
{
    float d = log(v->data);
    auto out = make_shared<Value>(d, vector{v});

    out->setBackward([d](shared_ptr<Value> &self)
                     { 
                        auto a = *(self->prev).begin();
                        if (a->data != 0) {
            a->grad += (1 / a->data) * self->grad;
        } else {
            cerr << "Gradient computation for log with data = 0." << endl;
        } });
    out->l = "log(" + v->l + ")";
    return out;
}

shared_ptr<Value> tanh(shared_ptr<Value> v)
{
    float d = std::tanh(v->data);
    shared_ptr out = make_shared<Value>(d, vector{v});
    out->l = "tanH(" + v->l + ")";
    out->setBackward([d](shared_ptr<Value> &self)
                     { 
                        auto v = *(self->prev).begin();;
                        
                        v->grad += (1 - d * d) * self->grad; });
    return out;
}
shared_ptr<Value> relu(shared_ptr<Value> v)
{
    float data = v->data;
    auto out = make_shared<Value>((data + abs(data)) / 2, vector{v});
    out->setBackward([data](shared_ptr<Value> &self)
                     { 
                        auto v = *(self->prev).begin();;
                        
                        v->grad += (data > 0) * self->grad; });
    out->l = "relu(" + v->l + ")";
    return out;
}

void Value::backward()
{
    vector<shared_ptr<Value>> topo = {};
    set<shared_ptr<Value>> visited = {};
    grad = 1;

    build_topo(shared_from_this(), topo, visited);

    for (int i = topo.size() - 1; i >= 0; i--)
    {

        topo[i]->_backward(topo[i]);
    }
}

ostream &operator<<(ostream &out, Value &v)
{
    out << v.l << "\t|" << v.data << "\t| grad = " << v.grad;
    return out;
}

void build_topo(const shared_ptr<Value> &v, vector<shared_ptr<Value>> &topo, set<shared_ptr<Value>> &visited)
{

    if (visited.find(v) == visited.end())
    {
        visited.insert(v);
        for (auto c : v->prev)
        {
            build_topo(c, topo, visited);
        }
        topo.push_back(v);
    }
}
