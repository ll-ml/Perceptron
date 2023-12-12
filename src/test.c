#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// Include your tensor.h here
#include "tensor.h"

void test_tensors() {
    int shape1[] = {4};
    Tensor* vec1 = tensor_rand(shape1, VECTOR);

    int shape2[] = {4};
    Tensor* vec2 = tensor_rand(shape2, VECTOR);

    Tensor* result = dot(vec1, vec2);
    print_tensor(result);

}

int main() {
    test_tensors();

    puts("\n\nAll tests passed!");

    return 0;
}
