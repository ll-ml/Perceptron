#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "tensor.h"
#include "activations.h"

void test_tensors() {
    //Matrix
    int shape[] = {10, 10};

    Tensor* t = tensor_rand(shape, 2);

    print_tensor(t);
    
    puts("----------------------------------------------------------------\n");

    relu_tensor(t);
    print_tensor(t);
}

int main() {
    test_tensors();

    puts("\n\nAll tests passed.");

    return 0;
}
