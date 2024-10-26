# cpp-micrograd

This project is a simple yet extensible implementation of a neural network in C++. It is inspired by [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy, which demonstrates the basics of automatic differentiation and backpropagation. This is a c++ implementation of the algorithm.

## Overview

The code implements a basic structure for creating and training neural networks, focusing on key components such as:

- **Neurons**: Individual units that perform computations and apply activation functions.
- **Layers**: Collections of neurons, including linear layers with specified input and output sizes.
- **Multi-Layer Perceptron (MLP)**: A simple architecture for creating neural networks composed of multiple layers.

## File Structure

```
./build
./examples
    ├── example1.cpp            // A basic demonstration of the library
./include
    ├── NN.hpp                  // Header file for neural network classes
    └── ValueStruct.hpp         // Header file for Value class (represents data and gradients)
./lib
    ├── NN.cpp                  // Implementation of neural network classes
    └── ValueStruct.cpp         // Implementation of the Value class
./tests
    ├── NN.test.cpp             // Tests for neural network classes
    └── ValueStruct.test.cpp    // Tests for the Value class
Makefile
```

## Key Classes

### Module

An abstract base class for neural network modules, defining the interface for obtaining parameters and zeroing gradients.

### Neuron

Represents a single neuron in the network. It can:

- Initialize with random weights or from a given set of parameters.
- Perform a forward pass using specified activation functions (e.g., ReLU, Tanh).
- Save and load its state.

### LinearLayer

Represents a layer of neurons in the network. It can:

- Forward propagate input through its neurons.
- Retrieve parameters for training.
- Save and load its state.

### MLP

Represents a multi-layer perceptron, allowing for the creation of neural networks with multiple layers. It can:

- Forward propagate input through all layers.
- Retrieve all parameters for training.
- Save and load the entire network's state.

## Functions

- **softMax**: Applies the softmax function to a vector of values.
- **simpleLoss**: Computes the mean squared error between predicted values and true labels.

## Usage

To use this project, include the `NN.hpp` and `ValueStruct.hpp` headers in your C++ files. You can create an MLP instance, add layers, and feed input data to perform forward passes. Gradients can be computed via backpropagation using the `backward()` method on the output values.

### Example

See `./examples/example1.cpp`

## Requirements

- C++11 or later
- A compatible C++ compiler (e.g., g++, clang++)

## Contributing

Contributions are welcome! Feel free to submit issues, feature requests, or pull requests.

## Acknowledgments

This implementation is inspired by [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy, which provides a minimalistic approach to understanding automatic differentiation and neural networks.

# Future

This is the first of 3 planned projects. I am currently working on a implementation of a backpropagation engine with using float arrays as the basic data type (Instead of singular values as in this project).
The third version will be an extension of the 2nd version, and will include CUDA code for faster computations.
