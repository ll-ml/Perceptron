#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "tensor.h"
#include "activations.h"

static const int TEST_NUM = 10000;

void test_tensors() {
    
    for (int i = 0; i < TEST_NUM; i++) {
        int t1_shape[] = {28, 28};
        Tensor* t1 = tensor_rand(t1_shape, MATRIX);

        int t2_shape[] = {28, 28};
        Tensor* t2 = tensor_rand(t2_shape, MATRIX);

        Tensor* result_tensor = dot(t1, t2);
        print_tensor(result_tensor);
        free_tensor(t1);
        free_tensor(t2); 
        free_tensor(result_tensor);
        puts("\n");
    }

}

int main() {
    test_tensors();

    puts("\n\nAll tests passed.");

    return 0;
}
