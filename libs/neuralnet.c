#include "headers/neuralnet.h"
#include <assert.h>
#include <stdlib.h>

struct Neuron *Neuron_create(int num_inputs) {

  struct Neuron *neuron = malloc(sizeof(struct Neuron));
  assert(neuron != NULL);

  neuron->num_inputs = num_inputs;

  neuron->bias = (double)rand() / RAND_MAX * 2.0 - 1.0;

  neuron->inputs = malloc(num_inputs * sizeof(double));
  assert(neuron->inputs != NULL);

  neuron->weights = malloc(num_inputs * sizeof(double));
  assert(neuron->weights != NULL);

  for (int i = 0; i < num_inputs; i++) {
    neuron->weights[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
    neuron->inputs[i] = 0;
  }

  return neuron;
}

void Neuron_destroy(struct Neuron *neuron) {

  assert(neuron != NULL);

  assert(neuron->inputs != NULL);
  free(neuron->inputs);

  assert(neuron->weights != NULL);
  free(neuron->weights);

  free(neuron);
}
