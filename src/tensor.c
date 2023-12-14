// tensor.c
#include "tensor.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

Tensor* dot(Tensor* t1, Tensor* t2) {
    Tensor* result = NULL;

    if (t1->num_dimensions == VECTOR && t2->num_dimensions == VECTOR) {
        if (t1->shape[0] != t2->shape[0]) {
            fprintf(stderr, "shapes (%d, ) and (%d, ) not aligned: %d (dim 0) != %d (dim 0)", t1->shape[0], t2->shape[0], t1->shape[0], t2->shape[0]);
            return result;
        }

        int shape[] = {1};
        result = create_tensor(shape, 1);
        float output = 0.0F;

        for (int i = 0; i < t1->shape[0]; i++) {
            output+= t1->data[0][i] * t2->data[0][i];
        }
        result->data[0][0] = output;
        return result;
    }

    if (t1->num_dimensions == MATRIX && t2->num_dimensions == VECTOR) {
        if (t1->shape[1] != t2->shape[0]) {
            fprintf(stderr, "shapes (%d, ) and (%d, ) not aligned: %d (dim 0) != %d (dim 0)", t1->shape[0], t2->shape[0], t1->shape[0], t2->shape[0]);
            return result;
        }
        int shape[] = {t1->shape[1]}; 
        result = create_tensor(shape, VECTOR);

    }

    /*
    if (t1->shape[0] != t2->shape[1]) {
        fprintf(stderr, "shape error dim0 != dim1 (%d != %d)\n", t1->shape[0], t2->shape[1]);
    }
    */
    return result;
}

float random_gamma(float shape, float scale) {
    srand((unsigned int)time(NULL) + (unsigned int)clock());
    float b, c;
    float U, V, X, Y, Z, S, D;
    
    b = shape - 1.0 / 3.0;
    c = 1.0 / sqrt(9.0 * b);
    
    do {
        do {
            // Generate two independent random variables U and V from a uniform distribution in (0, 1)
            U = (float)rand() / RAND_MAX;
            V = (float)rand() / RAND_MAX;
            
            X = c * (6.0F * U - 3.0F);
            Y = fabs(c * V);
        } while (fabs(X) > 1.0F || Y > 1.0F);

        Z = X * X * X;
        S = 1.0 + 0.33267F * X + 0.06377F * Z;
        D = 1.0 - exp(-X * X / 2.0) / sqrt(2.0F * M_PI);
    } while (rand() / (float)RAND_MAX > D && Y > 1.0F / (2.0F * S));

    return b * Z * scale;
}

Tensor* tensor_rand(int shape[], int num_dimensions) {
    Tensor* t = create_tensor(shape, num_dimensions);

    switch(t->num_dimensions) {
        case VECTOR:
            for (int i = 0; i < t->shape[0]; i++) {
                t->data[0][i] = (float)random_gamma(22.0, 22.0);
            }
            break;
        case MATRIX:
            for (int i = 0; i < t->shape[0]; i++) {
                for (int j = 0; j < t->shape[1]; j++) {
                    t->data[i][j] = (float)random_gamma(2.0, 1.0);
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

void print_tensor(Tensor* tensor) {
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
