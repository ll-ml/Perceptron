#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// Include your tensor.h here
#include "tensor.h"

void test_tensors() {
    int shape1[] = {3, 3};
    Tensor* matrix = tensor_rand(shape1, MATRIX);

    int shape2[] = {3};
    Tensor* vector = tensor_rand(shape2, VECTOR);

    Tensor* result = dot(matrix, vector);
    assert(result != NULL);
    print_tensor(result);
}

int main() {
    test_tensors();

    puts("\n\nAll tests passed.");

    return 0;
}
