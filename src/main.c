#include "../lib/headers/neuralnet.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {

  srand(time(NULL));

  int num_layers = 2;
  int num_neurons = 4;
  int num_neuron_inputs = 8;

  struct Layer *layers[num_layers];

  int l = 0;
  int nn = 0;
  int nni = 0;

  for (l = 0; l < num_layers; l++) {

    layers[l] = Layer_create(num_neurons, num_neuron_inputs);

    for (nn = 0; nn < num_neurons; nn++) {

      printf("\n\tlayers[%d]->neurons[%d]->bias:\t\t%9f\n\n", l, nn,
             layers[l]->neurons[nn]->bias);

      for (nni = 0; nni < num_neuron_inputs; nni++) {
        printf("\tlayers[%d]->neurons[%d]->weights[%d]:\t%9f\n", l, nn, nni + 1,
               layers[l]->neurons[nn]->weights[nni]);
      }
      printf("\n");

      for (nni = 0; nni < num_neuron_inputs; nni++) {
        printf("\tlayers[%d]->neurons[%d]->inputs[%d]:\t%9f\n", l, nn, nni + 1,
               layers[l]->neurons[nn]->inputs[nni]);
      }
      printf("\n");
    }
  }

  for (l = 0; l < num_layers; l++) {
    Layer_destroy(layers[l]);
  }

  return 0;
}
