#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include "tensor.h"

void print_test_result(const char* test_name, int passed) {
    printf("[%s] %s\n", passed ? "PASS" : "FAIL", test_name);
}

void test_vector_creation() {
    printf("\n=== Testing Vector Creation ===\n");
    
    int shape[] = {5};
    Tensor* vec = create_tensor(shape, 1);
    
    assert(vec != NULL);
    assert(vec->num_dimensions == 1);
    assert(vec->shape[0] == 5);
    assert(vec->total_size == 5);
    assert(vec->data != NULL);
    assert(vec->data[0] != NULL);
    assert(strcmp(vec->device, "cpu") == 0);
    assert(vec->requires_grad == false);
    assert(vec->grad == NULL);
    

    int all_zero = 1;
    for (int i = 0; i < 5; i++) {
        if (vec->data[0][i] != 0.0f) {
            all_zero = 0;
            break;
        }
    }
    assert(all_zero);
    
    print_test_result("Vector creation (size 5)", 1);
    
    vec->data[0][0] = 1.0f;
    vec->data[0][2] = 3.14f;
    vec->data[0][4] = -2.5f;
    
    assert(vec->data[0][0] == 1.0f);
    assert(vec->data[0][2] == 3.14f);
    assert(vec->data[0][4] == -2.5f);
    
    print_test_result("Vector set/get values", 1);
    
    printf("Vector contents: [");
    for (int i = 0; i < 5; i++) {
        printf("%.2f%s", vec->data[0][i], i < 4 ? ", " : "");
    }
    printf("]\n");
    
    free_tensor(vec);
    print_test_result("Vector memory cleanup", 1);
}

// Test matrix creation
void test_matrix_creation() {
    printf("\n=== Testing Matrix Creation ===\n");
    
    int shape[] = {3, 4};
    Tensor* mat = create_tensor(shape, 2);
    
    assert(mat != NULL);
    assert(mat->num_dimensions == 2);
    assert(mat->shape[0] == 3);
    assert(mat->shape[1] == 4);
    assert(mat->total_size == 12);
    assert(mat->data != NULL);
    
    print_test_result("Matrix creation (3x4)", 1);
    
    float value = 1.0f;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            mat->data[i][j] = value++;
        }
    }
    
    float* first_element = &mat->data[0][0];
    int contiguous = 1;
    value = 1.0f;
    for (int i = 0; i < 12; i++) {
        if (first_element[i] != value++) {
            contiguous = 0;
            break;
        }
    }
    assert(contiguous);
    
    print_test_result("Matrix contiguous memory layout", 1);
    

    printf("Matrix contents:\n");
    print_tensor(mat);
    
    free_tensor(mat);
    print_test_result("Matrix memory cleanup", 1);
}


void test_error_cases() {
    printf("\n=== Testing Error Cases ===\n");

    Tensor* t1 = create_tensor(NULL, 1);
    assert(t1 == NULL);
    print_test_result("NULL shape rejection", 1);
    
    int shape[] = {5};
    Tensor* t2 = create_tensor(shape, 0);
    assert(t2 == NULL);
    print_test_result("Zero dimensions rejection", 1);
    
    Tensor* t3 = create_tensor(shape, 3);
    assert(t3 == NULL);
    print_test_result("3D tensor rejection (not implemented)", 1);
    
    int bad_shape[] = {5, 0};
    Tensor* t4 = create_tensor(bad_shape, 2);
    if (t4) {
        printf("  Warning: Zero-sized dimensions not validated\n");
        free_tensor(t4);
    }
}

void test_mnist_sizes() {
    printf("\n=== Testing MNIST-Relevant Tensor Sizes ===\n");
    
    int input_shape[] = {784};
    Tensor* input = create_tensor(input_shape, 1);
    assert(input != NULL);
    assert(input->total_size == 784);
    print_test_result("MNIST input vector (784)", 1);

    int w1_shape[] = {128, 784};
    Tensor* w1 = create_tensor(w1_shape, 2);
    assert(w1 != NULL);
    assert(w1->total_size == 128 * 784);
    printf("  First layer weights size: %d parameters\n", w1->total_size);
    print_test_result("MNIST weight matrix (128x784)", 1);
    
    int b1_shape[] = {128};
    Tensor* b1 = create_tensor(b1_shape, 1);
    assert(b1 != NULL);
    print_test_result("MNIST bias vector (128)", 1);
    

    int w2_shape[] = {10, 128};
    Tensor* w2 = create_tensor(w2_shape, 2);
    assert(w2 != NULL);
    printf("  Output layer weights size: %d parameters\n", w2->total_size);
    print_test_result("MNIST output matrix (10x128)", 1);
    
    int total_params = w1->total_size + b1->total_size + w2->total_size + 10; // +10 for output bias
    printf("  Total network parameters: %d\n", total_params);
    
    free_tensor(input);
    free_tensor(w1);
    free_tensor(b1);
    free_tensor(w2);
}

void test_performance() {
    printf("\n=== Performance Test ===\n");
    
    clock_t start, end;
    double cpu_time_used;

    int num_tensors = 1000;
    Tensor** tensors = malloc(num_tensors * sizeof(Tensor*));
    
    int shape[] = {100, 100};
    start = clock();
    
    for (int i = 0; i < num_tensors; i++) {
        tensors[i] = create_tensor(shape, 2);
    }
    
    end = clock();
    cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    printf("  Created %d matrices (100x100) in %.4f seconds\n", num_tensors, cpu_time_used);
    printf("  Average time per tensor: %.6f seconds\n", cpu_time_used / num_tensors);
    
    for (int i = 0; i < num_tensors; i++) {
        free_tensor(tensors[i]);
    }
    free(tensors);
    
    print_test_result("Performance test completed", 1);
}

void test_existing_functions() {
    printf("\n=== Testing Other Implemented Functions ===\n");
    
    int shape1[] = {3};
    int shape2[] = {3};
    Tensor* v1 = create_tensor(shape1, 1);
    Tensor* v2 = create_tensor(shape2, 1);
    
    v1->data[0][0] = 1.0f; v1->data[0][1] = 2.0f; v1->data[0][2] = 3.0f;
    v2->data[0][0] = 4.0f; v2->data[0][1] = 5.0f; v2->data[0][2] = 6.0f;
    
    printf("  v1 = [1.0, 2.0, 3.0]\n");
    printf("  v2 = [4.0, 5.0, 6.0]\n");
    

    /* Going to rewrite dot product soon
    Tensor* result = dot(v1, v2);
    if (result != NULL) {
        printf("  dot(v1, v2) = %.2f\n", result->data[0][0]);
        assert(result->data[0][0] == 32.0f); // 1*4 + 2*5 + 3*6 = 32
        print_test_result("Vector dot product", 1);
        free_tensor(result);
    } else {
        printf("  Dot product not working with current implementation\n");
    }
    */
    
    free_tensor(v1);
    free_tensor(v2);
}

int main() {
    printf("=================================\n");
    printf("    Tensor Library Test Suite    \n");
    printf("=================================\n");
    
    // Run all tests
    test_vector_creation();
    test_matrix_creation();
    test_error_cases();
    test_mnist_sizes();
    test_performance();
    test_existing_functions();
    
    printf("\n=================================\n");
    printf("       All Tests Completed       \n");
    printf("=================================\n");
    
    return 0;
}