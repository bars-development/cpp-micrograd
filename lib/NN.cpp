#include "../include/NN.hpp"

void Module::zero_grad()
{
    {
        for (auto p : parameters())
        {
            p->setGrad(0);
        }
    }
}
// Neuron class definition
Neuron::Neuron(int nin, activation act) : act{act}
{
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    for (int i = 0; i < nin; i++)
    {
        w.push_back(make_shared<Value>(distribution(generator)));
    }
    b = make_shared<Value>(distribution(generator));
}
Neuron::Neuron(vector<float> params)
{
    act = activation(int(params[0]));
    for (int i = 1; i < params.size() - 1; i++)
    {
        w.push_back(make_shared<Value>(params[i]));
    }
    b = make_shared<Value>(params.back());
}

void Neuron::save(ostream &file)
{
    file << int(act) << endl;
    for (auto &weight : w)
    {
        file << weight->getData() << endl;
    }
    file << b->getData() << endl;
}

vector<shared_ptr<Value>> Neuron::parameters()
{
    auto c(w);
    c.push_back(b);
    return c;
}
shared_ptr<Value> Neuron::operator()(vector<shared_ptr<Value>> x)
{
    if (x.size() != w.size())
    {
        throw runtime_error("Input size does not match");
    }

    vector<shared_ptr<Value>> ws{};
    for (int i = 0; i < x.size(); i++)
    {
        ws.push_back(x[i] * w[i]);
    }
    ws.push_back(b);
    auto k = sum(ws);

    switch (act)
    {
    case activation::relu:
        return relu(k);
    case activation::tanh:
        return tanh(k);
    default:
        return k;
    }
}

// LinearLayer class definition
LinearLayer::LinearLayer(int nin, int nout, activation act) : nin{nin}, nout{nout}
{
    for (int i = 0; i < nout; i++)
    {
        Neuron n = Neuron(nin, act);
        neurons.push_back(n);
    }
}
LinearLayer::LinearLayer(istream &in)
{
    in >> nin >> nout;

    for (int i = 0; i < nout; i++)
    {
        vector<float> v = vector<float>(nin + 2);
        for (int j = 0; j < nin + 2; j++)
        {
            in >> v[j];
        }
        neurons.push_back(Neuron(v));
    }
}

void LinearLayer::save(ostream &file)
{
    file << nin << endl
         << nout << endl;
    for (auto &n : neurons)
    {
        n.save(file);
    }
}

vector<shared_ptr<Value>> LinearLayer::operator()(vector<shared_ptr<Value>> x)
{

    vector<shared_ptr<Value>> out{};
    for (int i = 0; i < nout; i++)
    {

        out.push_back(neurons[i](x));
    }

    return out;
}
vector<shared_ptr<Value>> LinearLayer::parameters()
{
    vector<shared_ptr<Value>> p{};
    for (int i = 0; i < neurons.size(); i++)
    {
        auto np = neurons[i].parameters();
        p.insert(p.end(), np.begin(), np.end());
    }
    return p;
}

// MLP class definition
MLP::MLP(int in, vector<int> l) : layers{{}}
{
    layers.push_back(LinearLayer(in, l[0], activation::tanh));
    for (int i = 0; i < l.size() - 1; i++)
    {
        LinearLayer layer = LinearLayer(l[i], l[i + 1], activation::tanh);

        layers.push_back(layer);
    }
}
MLP::MLP(string path)
{
    ifstream file(path);
    if (!file.is_open())
    {
        cerr << "Error opening file!" << endl;
        return;
    }
    string s;
    file >> s;
    int n;
    file >> n;

    for (int i = 0; i < n; i++)
    {

        layers.push_back(LinearLayer(file));
    }
    file.close();
}
void MLP::saveTo(string path)
{
    ofstream file(path, ios::out | ios::trunc);
    file << "MLP" << endl;
    file << layers.size() << endl;
    for (int i = 0; i < layers.size(); i++)
    {
        layers[i].save(file);
    }
    // cout << layers.size() << endl;
}

vector<shared_ptr<Value>> MLP::operator()(vector<std::shared_ptr<Value>> input)
{
    // vector<shared_ptr<Value>> x;
    // for (auto v : input)
    // {
    //     x.push_back(make_shared<Value>(v));
    // }
    auto x = input;
    for (int i = 0; i < layers.size(); i++)
    {

        x = layers[i](x);
    }
    return x;
}
vector<shared_ptr<Value>> MLP::parameters()
{
    vector<shared_ptr<Value>> p{};
    for (int i = 0; i < layers.size(); i++)
    {
        auto np = layers[i].parameters();
        p.insert(p.end(), np.begin(), np.end());
    }
    return p;
}

// softMax function definition
vector<shared_ptr<Value>> softMax(vector<shared_ptr<Value>> x)
{
    // shared_ptr<Value> s = make_shared<Value>(0);
    vector<shared_ptr<Value>> r;
    for (auto k : x)
    {
        r.push_back(exp(k));
        // s = s + r.back();
    }
    auto s = sum(r);
    for (int i = 0; i < r.size(); i++)
    {
        r[i] = r[i] / s;
    }
    return r;
}

// cross entropy loss function
shared_ptr<Value> simpleLoss(vector<shared_ptr<Value>> pred, vector<shared_ptr<Value>> y)
{
    auto loss = make_shared<Value>(0);
    if (pred.size() != y.size())
        throw runtime_error("pred and y different sizes");

    for (int i = 0; i < pred.size(); i++)
    {
        auto dif = (pred[i] - y[i]);
        loss = loss + (dif * dif);
    }
    loss = loss / make_shared<Value>(pred.size());
    return loss;
}