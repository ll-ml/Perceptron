// activations.h
#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

float relu(float x);
float sigmoid(float x);
float softmax(float* logits, int num_classes);

#endif // ACTIVATIONS_H
