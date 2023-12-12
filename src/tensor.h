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
    int num_dimensions; //possible rename to dims
    char* device;
} Tensor;

//Alloc + dealloc
Tensor* create_tensor(int shape[], int num_dimensions);
Tensor* tensor_rand(int shape[], int num_dimensions);
void free_tensor(Tensor* tensor);

//Ops
Tensor* dot(Tensor* t1, Tensor* t2);

//Other
void print_tensor(Tensor* tensor);
#endif // TENSOR_H
