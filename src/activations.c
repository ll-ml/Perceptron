// activations.c
#include "activations.h"
#include <math.h>

float relu(float x) {
    return (x > 0.0) ? x : 0.0;
}

float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

float softmax(float* logits, int num_classes) {
    // Implementation of softmax
    // ...
}
