// tensor.h
#ifndef TENSOR_H
#define TENSOR_H

//Currently only support 
#define VECTOR 1
#define MATRIX 2

//~28 bytes not accounting for padding/width diffs
typedef struct {
    float** data;
    int* shape;
    int num_dimensions;
    char* device;
} Tensor;

Tensor* create_tensor(int shape[], int num_dimensions);
void free_tensor(Tensor* tensor);
// Other tensor-related functions

#endif // TENSOR_H
