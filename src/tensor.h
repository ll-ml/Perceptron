// tensor.h
#ifndef TENSOR_H
#define TENSOR_H

#include <stdbool.h>

//Currently only support 
#define SCALAR 0
#define VECTOR 1
#define MATRIX 2
#define TENSOR3D 3 

typedef enum {
    INIT_ZEROS,
    INIT_ONEs,
    INIT_RANDOM_NORMAL,
    INIT_RANDOM_UNIFORM,
    INIT_XAVIER,
    INIT_HE
} InitType;

// Tensor
typedef struct {
    float** data;
    int* shape;
    int num_dimensions;
    int total_size;
    char* device;
    bool requires_grad;
    float** grad; 
} Tensor;

// Memory management
Tensor* create_tensor(int shape[], int num_dimensions);
Tensor* create_tensor_init(int shape[], int num_dimensions, InitType init_type);
Tensor* tensor_rand(int shape[], int num_dimensions);
Tensor* copy_tensor(const Tensor* src);
void free_tensor(Tensor* tensor);

//Ops
Tensor* add(const Tensor* t1, const Tensor* t2);
Tensor* subtract(const Tensor* t1, const Tensor* t2);
Tensor* multiply_elementwise(const Tensor* t1, const Tensor* t2);
Tensor* dot(const Tensor* t1, const Tensor* t2);
Tensor* transpose(const Tensor* t);
Tensor* reshape(const Tensor* t, int new_shape[], int new_dims);

// Scalar ops
Tensor* scalar_multiply(const Tensor* t, float scalar);
Tensor* scalar_add(const Tensor* t, float scalar);

// Activation functions
Tensor* relu(const Tensor* t);
Tensor* relu_backward(const Tensor* grad_output, const Tensor* input);
Tensor* sigmoid(const Tensor* t);
Tensor* sigmoid_backward(const Tensor* grad_output, const Tensor* output);
Tensor* softmax(const Tensor* t);

// Loss functions
float cross_entropy_loss(const Tensor* predictions, const Tensor* targets);
Tensor* cross_entropy_backward(const Tensor* predictions, const Tensor* targets);

// Utilities
void print_tensor(const Tensor* tensor);
void print_shape(const Tensor* tensor);
bool check_dimensions_match(const Tensor* t1, const Tensor* t2);
int get_total_size(int shape[], int num_dimensions);

typedef struct {
    Tensor* weights;
    Tensor* bias;
    Tensor* weight_grad;
    Tensor* bias_grad;
} Layer;

Layer* create_linear_layer(int input_size, int output_size);
Tensor* linear_forward(Layer* layer, const Tensor* input);

void linear_backward(Layer* layer, const Tensor* grad_output, const Tensor* input);
void update_weights(Layer* layer, float learning_rate);
void free_layer(Layer* layer);

#endif // TENSOR_H
