// activations.h
#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

double relu(double x);
double sigmoid(double x);
double softmax(double* logits, int num_classes);

#endif // ACTIVATIONS_H
