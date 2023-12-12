#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// Include your tensor.h here
#include "tensor.h"

void test_tensors() {
    // Test case 1
    puts("Starting tensors test...\n");
    int shape1[] = {2};
    Tensor* tensor1 = create_tensor(shape1, 1);

    assert(tensor1 != NULL);
    assert(tensor1->data != NULL);
    assert(tensor1->shape != NULL);
    assert(tensor1->num_dimensions == 1);
    assert(tensor1->shape[0] == 2);
    assert(strcmp(tensor1->device, "cpu") == 0);

    free_tensor(tensor1);
    tensor1 = NULL;
    puts("CREATE VECTOR TEST: PASSED...");

    // Test case 2
    int shape2[] = {28, 28};
    Tensor* tensor2 = create_tensor(shape2, 2);

    assert(tensor2 != NULL);
    assert(tensor2->data != NULL);
    assert(tensor2->shape != NULL);
    assert(tensor2->num_dimensions == 2);
    assert(tensor2->shape[0] == 28);
    assert(tensor2->shape[1] == 28);
    assert(tensor2->data[0][0] == 0);
    assert(sizeof(tensor2->data[0][0]) == sizeof(float));
    assert(strcmp(tensor2->device, "cpu") == 0);

    free_tensor(tensor2);
    tensor2 = NULL;
    puts("CREATE MATRIX TEST: PASSED...\n");

    int shape3[] = {20};
    Tensor* tensor3 = tensor_rand(shape3, 1);
    if (!tensor3) {
        fprintf(stderr, "ERROR creating tensor\n");
        return;
    }
    printf("Random vector shape(%d): \n", tensor3->shape[0]);
    printf("[ ");

    for (int i = 0; i < tensor3->shape[0]; i++) {
        printf("%.2f ", tensor3->data[0][i]);
    }
    printf("]");
    free(tensor3);
}

int main() {
    test_tensors();

    printf("\n\nAll tests passed!\n");

    return 0;
}
