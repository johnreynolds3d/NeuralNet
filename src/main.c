#include "../lib/headers/neuralnet.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {

  srand(time(NULL));

  int num_layers = 1;
  int num_neurons = 2;
  int num_neuron_inputs = 4;

  struct Layer *layers[num_layers];

  int l, nn, nni;

  for (l = 0; l < num_layers; l++) {

    layers[l] = Layer_create(num_neurons, num_neuron_inputs);

    for (nn = 0; nn < num_neurons; nn++) {

      printf("\nlayers[%d]->neurons[%d]->bias: %f\n\n", l + 1, nn + 1,
             layers[l]->neurons[nn]->bias);

      for (nni = 0; nni < num_neuron_inputs; nni++) {
        printf("layers[%d]->neurons[%d]->weights[%d]: %f\n", l + 1, nn + 1,
               nni + 1, layers[l]->neurons[nn]->weights[nni]);
      }
      printf("\n");

      for (nni = 0; nni < num_neuron_inputs; nni++) {
        printf("layers[%d]->neurons[%d]->inputs[%d]: %f\n", l + 1, nn + 1,
               nni + 1, layers[l]->neurons[nn]->inputs[nni]);
      }
      printf("\n");
    }
  }

  for (l = 0; l < num_layers; l++) {
    Layer_destroy(layers[l]);
  }

  return 0;
}
