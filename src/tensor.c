// tensor.c
#include "tensor.h"
#include <stdlib.h>
#include <string.h>

Tensor* create_tensor(int shape[], int num_dimensions) {
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    if (tensor == NULL) {
        return NULL;
    }

    int total_size = 1;
    for(int i = 0; i < num_dimensions; i++) {
        total_size *= shape[i];
    }

    switch (num_dimensions) {
        case VECTOR:
            tensor->data = (float**)malloc(sizeof(float*));
            tensor->data[0] = (float*)calloc(total_size, sizeof(float));
            break;
        case MATRIX:
            tensor->data = (float**)malloc(shape[0] * sizeof(float*));
            for(int i = 0; i < shape[0]; i++) {
                tensor->data[i] = (float*)calloc(shape[1], sizeof(float));
            }
            break;
    }

    //Here we use calloc and take a slight performance hit. Could use malloc + memset alternatively.
    //tensor->data = (float*)malloc(sizeof(float) * total_size);
    /*
    tensor->data = (float*)calloc(total_size, sizeof(float));
    if (tensor->data == NULL) {
        free(tensor->data);
        return NULL;
    }
    */

    tensor->shape = (int*)malloc(sizeof(int) * num_dimensions);
    if (tensor->shape == NULL) {
        free(tensor->data);
        free(tensor->shape);
        free(tensor);
        return NULL;
    }

    for(int i = 0; i < num_dimensions; i++) {
        tensor->shape[i] = shape[i];
    }

    tensor->num_dimensions = num_dimensions;
    
    tensor->device = (char*)malloc(strlen("cpu") + 1);
    if (tensor->device == NULL) {
        free(tensor->data);
        free(tensor->shape);
        free(tensor->device);
        free(tensor);
        return NULL;
    }

    strcpy(tensor->device, "cpu");
    return tensor;
}

void free_tensor(Tensor* tensor) {
    free(tensor->data);
    free(tensor->shape);
    free(tensor->device);
    free(tensor);
}

// Implement other tensor-related functions
