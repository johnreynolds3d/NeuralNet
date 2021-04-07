#include "headers/neuralnet.h"
#include <assert.h>
#include <stdlib.h>

Neuron *Neuron_create(int num_inputs) {

  Neuron *neuron = calloc(1, sizeof(Neuron));
  assert(neuron != NULL);

  neuron->num_inputs = num_inputs;

  neuron->bias = (double)rand() / RAND_MAX * 2.0 - 1.0;

  neuron->inputs = calloc(num_inputs, sizeof(double));
  assert(neuron->inputs != NULL);

  neuron->weights = calloc(num_inputs, sizeof(double));
  assert(neuron->weights != NULL);

  for (int i = 0; i < num_inputs; i++) {
    neuron->weights[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
    neuron->inputs[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
  }

  return neuron;
}

Layer *Layer_create(int num_neurons, int num_neuron_inputs) {

  Layer *layer = calloc(1, sizeof(Layer));
  assert(layer != NULL);

  layer->num_neurons = num_neurons;

  layer->neurons = calloc(num_neurons, sizeof(Neuron));
  assert(layer->neurons != NULL);

  for (int i = 0; i < num_neurons; i++) {
    layer->neurons[i] = Neuron_create(num_neuron_inputs);
  }

  return layer;
}

NeuralNet *NeuralNet_create(int num_inputs, int num_outputs,
                            int num_hidden_layers,
                            int num_neurons_per_hidden_layer, double alpha) {

  NeuralNet *neural_net = calloc(1, sizeof(NeuralNet));
  assert(neural_net != NULL);

  neural_net->num_inputs = num_inputs;
  neural_net->num_outputs = num_outputs;
  neural_net->num_hidden_layers = num_hidden_layers;
  neural_net->num_neurons_per_hidden_layer = num_neurons_per_hidden_layer;
  neural_net->alpha = alpha;

  if (num_hidden_layers > 0) {

    neural_net->layers = calloc(num_hidden_layers, sizeof(Layer));
    assert(neural_net->layers != NULL);

    neural_net->layers[0] =
        Layer_create(num_neurons_per_hidden_layer, num_inputs);

    for (int i = 0; i < num_neurons_per_hidden_layer - 1; i++) {
      neural_net->layers[i] = Layer_create(num_neurons_per_hidden_layer,
                                           num_neurons_per_hidden_layer);
    }

    neural_net->layers[num_neurons_per_hidden_layer] =
        Layer_create(num_outputs, num_neurons_per_hidden_layer);

  } else {

    neural_net->layers = malloc(sizeof(Layer));
    assert(neural_net->layers != NULL);

    neural_net->layers[0] = Layer_create(num_outputs, num_inputs);
  }

  return neural_net;
}

void Neuron_destroy(Neuron *neuron) {

  assert(neuron != NULL);

  assert(neuron->inputs != NULL);
  free(neuron->inputs);

  assert(neuron->weights != NULL);
  free(neuron->weights);

  free(neuron);
}

void Layer_destroy(Layer *layer) {

  assert(layer != NULL && layer->neurons != NULL);

  for (int i = 0; i < layer->num_neurons; i++) {
    Neuron_destroy(layer->neurons[i]);
  }

  free(layer->neurons);

  free(layer);
}
