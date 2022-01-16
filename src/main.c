#include "../lib/headers/neuralnet.h"
#include <stdint.h>

/*
 * Gaze in disbelief as an Artificial Neural Network (AKA Multilayer Perceptron)
 * does its darndest to learn, and perform. the logical operations AND, NAND,
 * OR, NOR, XOR, and XNOR!
 *
 * Author:  John Reynolds
 * Version: 16.01.22
 */

int main() {

  for (uint_fast8_t i = 0; i < 6; i++) {
    PreTraining(i);
  }

  return 0;
}
