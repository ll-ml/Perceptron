// activations.c
#include "activations.h"
#include "tensor.h"
#include <math.h>

float relu(float x) {
    return (x > 0.0) ? x : 0.0;
}

void relu_tensor(Tensor* t) {
    switch(t->num_dimensions) {
        case VECTOR:
            for (int i = 0; i < t->shape[0]; i++) {
                t->data[0][i] = (float)t->data[0][i] > 0 ? t->data[0][i] : 0.0F;
            }
            break;
        case MATRIX:
             for (int i = 0; i < t->shape[0]; i++) {
                for (int j = 0; j < t->shape[1]; j++) {
                    t->data[i][j] = (float)t->data[i][j] > 0 ? t->data[i][j] : 0.0F;
                }
             }
             break;
    }
}

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

void sigmoid_tensor(Tensor* t) {
    switch(t->num_dimensions) {
        case VECTOR:
            for (int i = 0; i < t->shape[0]; i++) {
                t->data[0][i] = (float)(1.0F) / (1.0F + exp(-t->data[0][i]));
            }
            break;
        case MATRIX:
            for (int i = 0; i < t->shape[0]; i++) {
                for (int j = 0; j < t->shape[1]; j++) {
                    t->data[i][j] = (float)(1.0F) / (1.0F + exp(-t->data[i][j]));
                }
            }
    }
}

float f32_tanh(float x) {
    return (float)tanh(x);
}

void tanh_tensor(Tensor* t) {
    switch(t->num_dimensions) {
        case VECTOR:
            for (int i = 0; i < t->shape[0]; i++) {
                t->data[0][i] = f32_tanh(t->data[0][i]);
            }
            break;
        case MATRIX:
            for (int i = 0; i < t->shape[0]; i++) {
                for (int j = 0; j < t->shape[1]; j++) {
                    t->data[i][j] = f32_tanh(t->data[i][j]);
                }
            }
    }
}

float softmax(float* logits, int num_classes) {
    // Implementation of softmax
    // ...
    return 0.0F;
}
