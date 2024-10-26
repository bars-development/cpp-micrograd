#include "include/NN.hpp"
#include <iostream>
#include <vector>
#include <memory>
#include <random>

// Function to generate synthetic training data (e.g., for a simple linear regression)
std::vector<std::pair<std::vector<std::shared_ptr<Value>>, std::vector<std::shared_ptr<Value>>>> generate_training_data(int num_samples)
{
    std::vector<std::pair<std::vector<std::shared_ptr<Value>>, std::vector<std::shared_ptr<Value>>>> data;
    std::mt19937 rng(42);
    std::uniform_real_distribution<> dist(-1.0, 1.0);

    for (int i = 0; i < num_samples; ++i)
    {
        double x1 = dist(rng);
        double x2 = dist(rng);
        double y = 3.0 * x1 + 2.0 * x2; // Target is a simple linear combination of inputs

        std::vector<std::shared_ptr<Value>> inputs = {std::make_shared<Value>(x1), std::make_shared<Value>(x2)};
        std::vector<std::shared_ptr<Value>> targets = {std::make_shared<Value>(y)};

        data.push_back({inputs, targets});
    }

    return data;
}

int main()
{
    // Define the MLP architecture
    MLP model(2, {5, 1}); // 2 input features, 1 hidden layer with 5 units, 1 output

    // Generate synthetic training data
    int num_samples = 100;
    auto training_data = generate_training_data(num_samples);

    // Training loop
    int epochs = 10000;
    double learning_rate = 0.001;

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        auto total_loss = std::make_shared<Value>(0.0);

        for (const auto &sample : training_data)
        {
            const std::vector<std::shared_ptr<Value>> &inputs = sample.first;
            const std::vector<std::shared_ptr<Value>> &targets = sample.second;

            // Forward pass
            std::vector<std::shared_ptr<Value>> predictions = model(inputs);

            // Compute loss (mean squared error)
            auto loss = simpleLoss(predictions, targets);
            total_loss = total_loss + loss;

            // Backward pass
            model.zero_grad(); // Reset gradients
            loss->backward();  // Backpropagation

            // Update model parameters (simple SGD)
            for (auto &p : model.parameters())
            {
                p->setData(p->getData() - learning_rate * p->getGrad());
            }
        }

        // Print the loss for this epoch
        if (epoch % 100 == 0)
        {
            std::cout << "Epoch " << epoch << " | Loss: " << total_loss->getData() / training_data.size() << std::endl;
        }
    }

    // Validation on new data
    auto validation_data = generate_training_data(20); // Generate some validation data
    auto validation_loss = std::make_shared<Value>(0.0);

    for (const auto &sample : validation_data)
    {
        const std::vector<std::shared_ptr<Value>> &inputs = sample.first;
        const std::vector<std::shared_ptr<Value>> &targets = sample.second;

        // Forward pass
        std::vector<std::shared_ptr<Value>> predictions = model(inputs);

        // Compute validation loss
        auto loss = simpleLoss(predictions, targets);
        validation_loss = validation_loss + loss;
    }

    std::cout << "Validation Loss: " << validation_loss->getData() / 20. << std::endl;

    return 0;
}
