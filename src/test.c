#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

// Include your tensor.h here
#include "tensor.h"
#include "activations.h"

void test_tensors() {
    int shape[] = {28, 28}; 
    Tensor* t = tensor_rand(shape, MATRIX);
    print_tensor(t);
    tanh_tensor(t);
    puts("--------------------------------------------------------------------------------------------\n");
    print_tensor(t);
}

int main() {
    test_tensors();

    puts("\n\nAll tests passed.");

    return 0;
}
