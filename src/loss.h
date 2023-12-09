// loss.h
#ifndef LOSS_H
#define LOSS_H

double sparse_categorical_crossentropy(int true_label, double* logits, int num_classes);
// Other loss functions

#endif // LOSS_H