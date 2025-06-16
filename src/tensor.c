/*
***
Perceptron Simple MLP
2025 By: https://github.com/ll-ml/Perceptron
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
***
*/
#include "tensor.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

Tensor* create_tensor(int shape[], int num_dimensions) {
    if (shape == NULL || num_dimensions <= 0 || num_dimensions >= 3) {
        fprintf(stderr, "Invalid tensor parameters\n");
        return NULL;
    }

    for (int i = 0; i < num_dimensions; i++) {
        if (shape[i] <= 0) {
            fprintf(stderr, "Invalid shape: dimension %d has size %d (must be positive)\n", 
                    i, shape[i]);
            return NULL;
        }
    }

    Tensor* tensor = (Tensor*)calloc(1, sizeof(Tensor));
    if (tensor == NULL) {
        fprintf(stderr, "Failed to allocate tensor structure\n");
        return NULL;
    }

    tensor->total_size = get_total_size(shape, num_dimensions);

    tensor->shape = (int*)malloc(num_dimensions * sizeof(int));
    if (tensor->shape == NULL) {
        free(tensor);
        return NULL;
    }
    memcpy(tensor->shape, shape, num_dimensions * sizeof(int));
    tensor->num_dimensions = num_dimensions;

    if (num_dimensions == 1) {
        tensor->data = (float**)malloc(sizeof(float*)); // One float ptr as its a vector
        if (tensor->data == NULL) {
            free(tensor->shape);
            free(tensor);
            return NULL;
        }
        
        tensor->data[0] = (float*)calloc(shape[0], sizeof(float));
        if (tensor->data[0] == NULL) {
            free(tensor->data);
            free(tensor->shape);
            free(tensor);
            return NULL;
        }
    } else if (num_dimensions == 2) {
        tensor->data = (float**)malloc(sizeof(float*) * shape[0]); // grab first dim from shape
        if (tensor->data == NULL) {
            free(tensor->shape);
            free(tensor);
            return NULL;
        }

        float* data_block = (float*)calloc(tensor->total_size, sizeof(float));
        if (data_block == NULL) {
            free(tensor->data);
            free(tensor->shape);
            free(tensor);
            return NULL;
        }

        for (int i = 0; i < shape[0]; i++) {
            tensor->data[i] = data_block + i * shape[1];
        }
    }

    tensor->device = strdup("cpu");
    if (tensor->device == NULL) {
        if (tensor->data) {
            if (tensor->data[0]) {
                free(tensor->data[0]);
            }
            free(tensor->data[0]);
        }
        free(tensor->shape);
        free(tensor);
        return NULL;
    }
    tensor->requires_grad = false;
    tensor->grad = NULL;

    return tensor;
}

int get_total_size(int shape[], int num_dimensions) {
    int total_size = 1;

    for (int i = 0; i < num_dimensions; i++) {
        total_size *= shape[i];
    }

    return total_size;
}

void print_tensor(const Tensor* tensor) {
    switch (tensor->num_dimensions) {
        case VECTOR: 
            for (int i = 0; i < tensor->shape[0]; i++) {
                printf("%.2f ", tensor->data[0][i]);
            }
            puts("\n");
            break;
        case MATRIX:
            for (int i = 0; i < tensor->shape[0]; i++) {
                for (int j = 0; j < tensor->shape[1]; j++) {
                    printf("%.2f ", tensor->data[i][j]);
                }
                puts("\n");
            }
    }
}

void free_tensor(Tensor* tensor) {
    if (tensor == NULL) return;
    
    // Free the data
    if (tensor->data != NULL) {
        if (tensor->data[0] != NULL) {
            free(tensor->data[0]);  // Free the actual data
        }
        free(tensor->data);  // Free the pointer array
    }
    
    // Free gradient tensor if it exists
    if (tensor->grad != NULL) {
        free_tensor(tensor->grad);
    }
    
    // Free other fields
    if (tensor->shape != NULL) {
        free(tensor->shape);
    }
    
    if (tensor->device != NULL) {
        free(tensor->device);
    }
    
    // Finally free the tensor structure itself
    free(tensor);
}