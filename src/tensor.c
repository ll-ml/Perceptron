// tensor.c
#include "tensor.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>


Tensor* tensor_rand(int shape[], int num_dimensions) {
    srand(time(NULL)); // TODO: fix
    Tensor* t = create_tensor(shape, num_dimensions);

    switch(t->num_dimensions) {
        case VECTOR:
            for (int i = 0; i < t->shape[0]; i++) {
                t->data[0][i] = (float)rand()/(float)(RAND_MAX) * 5.0;
            }
            break;
        case MATRIX:
            for (int i = 0; i < t->shape[0]; i++) {
                for (int j = 0; j < t->shape[1]; j++) {
                    t->data[i][j] = (float)rand()/(float)(RAND_MAX) * 5.0;
                }
            }
            break;
    }

    return t;
}


Tensor* create_tensor(int shape[], int num_dimensions) {
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    if (tensor == NULL) {
        return NULL;
    }

    switch (num_dimensions) {
        case VECTOR:
            tensor->data = (float**)malloc(sizeof(float*));
            tensor->data[0] = (float*)calloc(shape[0], sizeof(float));
            break;
        case MATRIX:
            tensor->data = (float**)malloc(shape[0] * sizeof(float*));
            for(int i = 0; i < shape[0]; i++) {
                tensor->data[i] = (float*)calloc(shape[1], sizeof(float));
            }
            break;
    }

    tensor->shape = (int*)calloc(num_dimensions, sizeof(int));
    if (tensor->shape == NULL) {
        free_tensor(tensor);
        return NULL;
    }

    for(int i = 0; i < num_dimensions; i++) {
        tensor->shape[i] = shape[i];
    }

    tensor->num_dimensions = num_dimensions;
    
    tensor->device = (char*)malloc(strlen("cpu") + 1);
    if (tensor->device == NULL) {
        free_tensor(tensor);
        return NULL;
    }

    strcpy(tensor->device, "cpu");
    return tensor;
}

void free_tensor(Tensor* tensor) {
    switch (tensor->num_dimensions) {
        case VECTOR:
            free(tensor->data[0]);
            free(tensor->data);
            break;
        case MATRIX:
            for (int i = 0; i < tensor->shape[0]; i++) {
                free(tensor->data[i]);
            }
            free(tensor->data);
            break;
    }

    free(tensor->shape);
    free(tensor->device);
    free(tensor);
}
