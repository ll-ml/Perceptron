// loss.h
#ifndef LOSS_H
#define LOSS_H

float sparse_categorical_crossentropy(int true_label, float* logits, int num_classes);
// Other loss functions

#endif // LOSS_H