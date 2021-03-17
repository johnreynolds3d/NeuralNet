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
    neuron->inputs[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
  }

  return neuron;
}

struct Layer *Layer_create(int num_neurons, int num_neuron_inputs) {

  struct Layer *layer = malloc(sizeof(struct Layer));
  assert(layer != NULL);

  layer->num_neurons = num_neurons;

  layer->neurons = malloc(num_neurons * sizeof(struct Neuron));
  assert(layer->neurons != NULL);

  for (int i = 0; i < num_neurons; i++) {
    layer->neurons[i] = Neuron_create(num_neuron_inputs);
  }

  return layer;
}

void Neuron_destroy(struct Neuron *neuron) {

  assert(neuron != NULL);

  assert(neuron->inputs != NULL);
  free(neuron->inputs);

  assert(neuron->weights != NULL);
  free(neuron->weights);

  free(neuron);
}

void Layer_destroy(struct Layer *layer) {

  assert(layer != NULL);

  assert(layer->neurons != NULL);
  for (int i = 0; i < layer->num_neurons; i++) {
    Neuron_destroy(layer->neurons[i]);
  }
  free(layer->neurons);

  free(layer);
}
