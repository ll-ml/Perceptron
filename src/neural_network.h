// neural_network.h
#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "tensor.h"
#include "activations.h"
#include "loss.h"
#include "optimizer.h"

typedef struct {
    // Definition of your neural network structure
    // ...
} NeuralNetwork;

NeuralNetwork* create_neural_network(/* parameters */);
void train_neural_network(NeuralNetwork* network, /* training data */);
double evaluate_neural_network(NeuralNetwork* network, /* evaluation data */);

#endif // NEURAL_NETWORK_H
