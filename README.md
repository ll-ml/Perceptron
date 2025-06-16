# Perceptron
### Ultra simple machine learning library.

 The overall goal of this project is to develop a deep fundamental understanding of how neural networks actually work. In order to achieve this I am writing this library _Perceptron_ which will be a standard multilayer perceptron library with backpropagation all implemented from scratch in C. 

### Tensor.c improvements:
This is a pretty cool memory hack to essentially get the standard C 2D array indexing without the poor cache locality it generally comes with.

**Before** 
Tensors would be allocated via a standard malloc call for a float** where each float pointer in the array would then point to some other pointer off in heap memory where each row would live. 
This was a nice API but very fragmented. See below:

data[0] → [0x1000] → [1.0][2.0][3.0][4.0] ← Separate allocation<br>
data[1] → [0x2000] → [5.0][6.0][7.0][8.0] ← Could be anywhere!<br>
data[2] → [0x3000] → [9.0][10.0][11.0][12.0] ← Poor cache locality

**Now** 
Now I allocate one big data block and use some basic pointer arithmetic to have each row actually be made up of the memory block so it is contiguous in memory. 

Memory Layout:<br>
Address: 0x1000  0x1004  0x1008  0x100C  0x1010  0x1014  0x1018  0x101C  0x1020  0x1024  0x1028  0x102C<br>
Values:  [  1.0 ][  2.0 ][  3.0 ][  4.0 ][  5.0 ][  6.0 ][  7.0 ][  8.0 ][  9.0 ][ 10.0 ][ 11.0 ][ 12.0 ]<br>
Index:      [0]     [1]     [2]     [3]     [4]     [5]     [6]     [7]     [8]     [9]    [10]    [11]<br>
        |------------ Row 0 ------------|------------ Row 1 ------------|------------ Row 2 ------------|

Row Pointers:<br>
data[0] = 0x1000 (points to index 0)  ─┐<br>
data[1] = 0x1010 (points to index 4)  ─┼─→ All point into the SAME memory block</br>
data[2] = 0x1020 (points to index 8)  ─┘
