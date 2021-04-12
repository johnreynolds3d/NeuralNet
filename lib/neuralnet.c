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
                            double learning_rate) {

  NeuralNet *neural_net = calloc(1, sizeof(NeuralNet));
  assert(neural_net != NULL);

  neural_net->num_inputs = num_inputs;
  neural_net->num_outputs = num_outputs;
  neural_net->num_hidden_layers = num_hidden_layers;
  neural_net->neurons_per_hidden_layer = neurons_per_hidden_layer;
  neural_net->learning_rate = learning_rate;

  neural_net->layers = calloc(
      (num_hidden_layers > 0) ? num_hidden_layers + 2 : 1, sizeof(Layer));
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

TrainingSet *TrainingSet_create(int num_inputs, double *inputs, int num_outputs,
                                double *desired_outputs) {

  assert(inputs != NULL && desired_outputs != NULL);

  TrainingSet *training_set = calloc(1, sizeof(TrainingSet));
  assert(training_set != NULL);

  training_set->inputs = calloc(num_inputs, sizeof(double));
  assert(training_set->inputs != NULL);

  training_set->desired_outputs = calloc(num_outputs, sizeof(double));
  assert(training_set->desired_outputs != NULL);

  int i;

  for (i = 0; i < num_inputs; i++) {
    training_set->inputs[i] = inputs[i];
  }

  for (i = 0; i < num_outputs; i++) {
    training_set->desired_outputs[i] = desired_outputs[i];
  }

  return training_set;
}

void Update_weights(NeuralNet *neural_net, double *desired_output,
                    double *outputs) {

  assert(neural_net != NULL && desired_output != NULL && outputs != NULL);

  double error = 0;

  int num_layers =
      neural_net->num_hidden_layers > 0 ? neural_net->num_hidden_layers + 2 : 1;

  int i, j, k, p;

  // loop through layers from last to first (backpropagation)
  for (i = num_layers - 1; i >= 0; i--) {

    // loop through neurons
    for (j = 0; j < neural_net->layers[i]->num_neurons; j++) {

      // if output layer, compute error
      if (i == num_layers - 1) {

        error = desired_output[j] - outputs[j];

        // error gradient calculated with delta rule
        neural_net->layers[i]->neurons[j]->error_gradient =
            outputs[j] * (1 - outputs[j]) * error;

      } else {

        neural_net->layers[i]->neurons[j]->error_gradient =
            neural_net->layers[i]->neurons[j]->output *
            (1 - neural_net->layers[i]->neurons[j]->output);

        double error_gradient_sum = 0;

        for (p = 0; p < neural_net->layers[i + 1]->num_neurons; p++) {
          error_gradient_sum +=
              neural_net->layers[i + 1]->neurons[p]->error_gradient *
              neural_net->layers[i + 1]->neurons[p]->weights[j];
        }

        neural_net->layers[i]->neurons[j]->error_gradient *= error_gradient_sum;
      }

      // loop through inputs
      for (k = 0; k < neural_net->layers[i]->neurons[j]->num_inputs; k++) {

        // if output layer, compute error
        if (i == num_layers - 1) {

          error = desired_output[j] - outputs[j];
          neural_net->layers[i]->neurons[j]->weights[k] +=
              neural_net->learning_rate *
              neural_net->layers[i]->neurons[j]->inputs[k] * error;

        } else {

          neural_net->layers[i]->neurons[j]->weights[k] +=
              neural_net->learning_rate *
              neural_net->layers[i]->neurons[j]->inputs[k] *
              neural_net->layers[i]->neurons[j]->error_gradient;
        }
      }

      neural_net->layers[i]->neurons[j]->bias +=
          neural_net->learning_rate * -1 *
          neural_net->layers[i]->neurons[j]->error_gradient;
    }
  }
}

double Step(double value) { return value < 0 ? 0 : 1; }

double Sigmoid(double value) {

  double k = exp(value);

  return k / (1.0f + k);
}

double BinaryStep(double value) { return value < 0 ? 0 : 1; }

double TanH(double value) { return 2 * Sigmoid(2 * value) - 1; }

double ReLu(double value) { return value > 0 ? value : 0; }

double LeakyReLu(double value) { return value < 0 ? 0.01 * value : value; }

double Sinusoid(double value) { return sin((float)value); }

double ArcTan(double value) { return atan((float)value); }

double SoftSign(double value) { return value / (1 + abs((float)value)); }

double Activation_function_output_layer(double value) { return Sigmoid(value); }

double Activation_function(double value) { return Sigmoid(value); }

void Train(NeuralNet *neural_net, TrainingSet *training_set, double *outputs) {

  assert(neural_net != NULL && training_set != NULL && outputs != NULL);

  int num_layers =
      neural_net->num_hidden_layers > 0 ? neural_net->num_hidden_layers + 2 : 1;

  double training_inputs[neural_net->num_inputs];

  int i, j, k;

  for (i = 0; i < neural_net->num_inputs; i++) {
    training_inputs[i] = training_set->inputs[i];
  }

  // loop through layers
  for (i = 0; i < num_layers; i++) {

    // if not input layer, set inputs to previous layer's outputs
    if (i > 0) {

      for (j = 0; j < neural_net->num_inputs; j++) {
        training_inputs[j] = outputs[0];
      }

      // clear outputs
      for (j = 0; j < neural_net->num_outputs; j++) {
        outputs[j] = 0;
      }
    }

    // loop through neurons
    for (j = 0; j < neural_net->layers[i]->num_neurons; j++) {

      for (k = 0; k < neural_net->layers[i]->neurons[j]->num_inputs; k++) {
        neural_net->layers[i]->neurons[j]->inputs[k] = 0;
      }

      double N = 0;

      // loop through inputs
      for (k = 0; k < neural_net->layers[i]->neurons[j]->num_inputs; k++) {

        neural_net->layers[i]->neurons[j]->inputs[k] = training_inputs[k];

        // compute dot product
        N += neural_net->layers[i]->neurons[j]->weights[k] * training_inputs[k];
      }

      // subtract bias
      N -= neural_net->layers[i]->neurons[j]->bias;

      // compute output for output layer
      if (i == num_layers - 1) {
        neural_net->layers[i]->neurons[j]->output =
            Activation_function_output_layer(N);
      } else { // compute output for hidden layers
        neural_net->layers[i]->neurons[j]->output = Activation_function(N);
      }

      outputs[i] = neural_net->layers[i]->neurons[j]->output;
    }
  }

  Update_weights(neural_net, training_set->desired_outputs, outputs);
}

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

  int num_layers =
      neural_net->num_hidden_layers > 0 ? neural_net->num_hidden_layers + 2 : 1;

  for (int i = 0; i < num_layers; i++) {
    Layer_destroy(neural_net->layers[i]);
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
