# Perceptron
## Ultra simple machine learning library.

 The overall goal of this project is to develop a deep fundamental understanding of how neural networks actually work. In order to achieve this I am writing this library _Perceptron_ which will be a standard multilayer perceptron library with backpropagation all implemented from scratch in C. 

### Tensor.c improvements:
This is a pretty cool memory hack to essentially get the standard C 2D array indexing without the poor cache locality it generally comes with.

**Before** 
Tensors would be allocated via a standard malloc call for a float** where each float pointer in the array would then point to some other pointer off in heap memory where each row would live. 
This was a nice API but very fragmented. See below:
data[0] → [0x1000] → [1.0][2.0][3.0][4.0]        ← Separate allocation
data[1] → [0x2000] → [5.0][6.0][7.0][8.0]        ← Could be anywhere!
data[2] → [0x3000] → [9.0][10.0][11.0][12.0]     ← Poor cache locality

**Now** 
Now I allocate one big data block and use some basic pointer arithmetic to have each row actually be made up of the memory block so it is contiguous in memory. 