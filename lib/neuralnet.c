#include "headers/neuralnet.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Neuron *Neuron_create(int num_inputs) {

  Neuron *neuron = calloc(1, sizeof(Neuron));
  assert(neuron != NULL);

  neuron->num_inputs = num_inputs;

  neuron->bias = (double)rand() / RAND_MAX * 2.0 - 1.0;

  neuron->output = 0;

  neuron->error_gradient = 0;

  neuron->inputs = calloc(num_inputs, sizeof(double));
  assert(neuron->inputs != NULL);

  neuron->weights = calloc(num_inputs, sizeof(double));
  assert(neuron->weights != NULL);

  for (int i = 0; i < num_inputs; i++) {
    neuron->weights[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
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
                            int num_hidden_layers, int neurons_per_hidden_layer,
                            double alpha) {

  NeuralNet *neural_net = calloc(1, sizeof(NeuralNet));
  assert(neural_net != NULL);

  neural_net->num_inputs = num_inputs;
  neural_net->num_outputs = num_outputs;
  neural_net->num_hidden_layers = num_hidden_layers;
  neural_net->neurons_per_hidden_layer = neurons_per_hidden_layer;
  neural_net->alpha = alpha;

  neural_net->layers = calloc(
      (num_hidden_layers < 0) ? num_hidden_layers + 2 : 1, sizeof(Layer));
  assert(neural_net->layers != NULL);

  if (num_hidden_layers > 0) {

    // create input layer
    neural_net->layers[0] = Layer_create(neurons_per_hidden_layer, num_inputs);

    // create hidden layers
    for (int i = 0; i < num_hidden_layers; i++) {
      neural_net->layers[i + 1] =
          Layer_create(neurons_per_hidden_layer, neurons_per_hidden_layer);
    }
    // create output layer
    neural_net->layers[num_hidden_layers + 1] =
        Layer_create(num_outputs, neurons_per_hidden_layer);

  } else {

    // create single layer
    neural_net->layers[0] = Layer_create(num_outputs, num_inputs);
  }

  return neural_net;
}

TrainingSet *TrainingSet_create(double *inputs, double *desired_outputs) {

  assert(inputs != NULL && desired_outputs != NULL);

  TrainingSet *training_set = calloc(1, sizeof(TrainingSet));
  assert(training_set != NULL);

  training_set->inputs =
      calloc(sizeof(inputs) / sizeof(inputs[0]), sizeof(inputs[0]));
  assert(training_set->inputs != NULL);

  training_set->desired_outputs =
      calloc(sizeof(desired_outputs) / sizeof(desired_outputs[0]),
             sizeof(desired_outputs[0]));
  assert(training_set->desired_outputs != NULL);

  training_set->inputs = inputs;
  training_set->desired_outputs = desired_outputs;

  return training_set;
}

void Update_weights(NeuralNet *neural_net, double *desired_output,
                    double *outputs) {

  assert(neural_net != NULL && desired_output != NULL && outputs != NULL);

  double error = 0;

  for (int i = neural_net->num_hidden_layers; i >= 0; i--) {

    for (int j = 0; j < neural_net->layers[i]->num_neurons; j++) {

      if (i == neural_net->num_hidden_layers) {

        error = desired_output[j] - outputs[j];

        // en.wikipedia.org/wiki/Delta_rule
        neural_net->layers[i]->neurons[j]->error_gradient =
            outputs[j] * (1 - outputs[j]) * error;

      } else {

        neural_net->layers[i]->neurons[j]->error_gradient =
            neural_net->layers[i]->neurons[j]->output *
            (1 - neural_net->layers[i]->neurons[j]->output);

        double error_gradient_sum = 0;

        for (int p = 0; p < neural_net->layers[i + 1]->num_neurons; p++) {
          error_gradient_sum +=
              neural_net->layers[i + 1]->neurons[p]->error_gradient *
              neural_net->layers[i + 1]->neurons[p]->weights[j];
        }

        neural_net->layers[i]->neurons[j]->error_gradient *= error_gradient_sum;
      }

      for (int k = 0; k < neural_net->layers[i]->neurons[j]->num_inputs; k++) {

        if (i == neural_net->num_hidden_layers) {

          error = desired_output[j] - outputs[j];
          neural_net->layers[i]->neurons[j]->weights[k] +=
              neural_net->alpha * neural_net->layers[i]->neurons[j]->inputs[k] *
              error;

        } else {

          neural_net->layers[i]->neurons[j]->weights[k] +=
              neural_net->alpha * neural_net->layers[i]->neurons[j]->inputs[k] *
              neural_net->layers[i]->neurons[j]->error_gradient;
        }
      }

      neural_net->layers[i]->neurons[j]->bias +=
          neural_net->alpha * -1 *
          neural_net->layers[i]->neurons[j]->error_gradient;
    }
  }
}

double Step(double value) { return value < 0 ? 0 : 1; }

double Sigmoid(double value) {

  double k = exp(value);

  return k / (1.0f + k);
}

double Activation_function(double value) { return Sigmoid(value); }

void Train(NeuralNet *neural_net, TrainingSet *training_set) {

  assert(neural_net != NULL && training_set != NULL);

  double inputs[2];
  double outputs[1];

  for (int i = 0; i < neural_net->num_hidden_layers + 1; i++) {

    if (i > 0) {
      inputs[0] = 0;
      inputs[1] = 0;
    }
    outputs[0] = 0;

    for (int j = 0; neural_net->layers[i]->num_neurons; j++) {

      double N = 0;
      neural_net->layers[i]->neurons[j]->inputs = 0;

      for (int k = 0; k < neural_net->layers[i]->neurons[j]->num_inputs; k++) {

        neural_net->layers[i]->neurons[j]->inputs[k] = inputs[k];
        N += neural_net->layers[i]->neurons[j]->weights[k] * inputs[k];
      }

      N -= neural_net->layers[i]->neurons[j]->bias;
      neural_net->layers[i]->neurons[j]->output = Activation_function(N);
      outputs[i] = neural_net->layers[i]->neurons[j]->output;
    }
  }

  Update_weights(neural_net, training_set->desired_outputs, outputs);
}

/*
void Train(NeuralNet *neural_net, double input_1, double input_2,
           double desired_output, double *inputs, double *outputs) {

  assert(neural_net != NULL && inputs != NULL && outputs != NULL);

  inputs[0] = input_1;
  inputs[1] = input_2;

  Go(neural_net, desired_output, inputs, outputs);
}
*/

void Neuron_destroy(Neuron *neuron) {

  assert(neuron != NULL);

  free(neuron->inputs);
  free(neuron->weights);

  free(neuron);
}

void Layer_destroy(Layer *layer) {

  assert(layer != NULL);

  for (int i = 0; i < layer->num_neurons; i++) {
    Neuron_destroy(layer->neurons[i]);
  }
  free(layer->neurons);

  free(layer);
}

void NeuralNet_destroy(NeuralNet *neural_net) {

  assert(neural_net != NULL);

  for (int i = 0; i < neural_net->num_hidden_layers; i++) {
    free(neural_net->layers[i]);
  }
  free(neural_net->layers);

  free(neural_net);
}

void TrainingSet_destroy(TrainingSet *training_set) {

  assert(training_set != NULL);

  free(training_set->inputs);
  free(training_set->desired_outputs);

  free(training_set);
}
