#include <stdio.h>

struct Neuron {
  int num_inputs;
  double bias;
  double output;
  double error_gradient;
  double weights[];
  double inputs[];
};

int main() { return 0 }
