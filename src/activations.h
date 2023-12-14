// activations.h
#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "tensor.h"

//Dense layers
float relu(float x);
void relu_tensor(Tensor* t);

float sigmoid(float x);
void sigmoid_tensor(Tensor* t);

float f32_tanh(float x);
void tanh_tensor(Tensor* t);

//Output layer
float softmax(float* logits, int num_classes);

#endif // ACTIVATIONS_H
