#include "headers/neuralnet.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Neuron *Neuron_create(int num_inputs) {

  Neuron *neuron = malloc(sizeof(Neuron));
  assert(neuron != NULL);

  neuron->num_inputs = num_inputs;
  neuron->bias = rand() / RAND_MAX * 2 - 1;
  neuron->output = 0;
  neuron->error_gradient = 0;

  neuron->inputs = calloc(num_inputs, sizeof(double));
  assert(neuron->inputs != NULL);

  neuron->weights = malloc(num_inputs * sizeof(double));
  assert(neuron->weights != NULL);

  for (int i = 0; i < num_inputs; i++) {
    neuron->weights[i] = rand() / RAND_MAX * 2 - 1;
  }

  return neuron;
}

Layer *Layer_create(int num_neurons, int num_neuron_inputs) {

  Layer *layer = malloc(sizeof(Layer));
  assert(layer != NULL);

  layer->num_neurons = num_neurons;

  layer->neurons = malloc(num_neurons * sizeof(Neuron));
  assert(layer->neurons != NULL);

  for (int i = 0; i < num_neurons; i++) {
    layer->neurons[i] = Neuron_create(num_neuron_inputs);
  }

  return layer;
}

NeuralNet *NeuralNet_create(int num_inputs, int num_outputs,
                            int num_hidden_layers, int neurons_per_hidden_layer,
                            double learning_rate) {

  NeuralNet *neural_net = malloc(sizeof(NeuralNet));
  assert(neural_net != NULL);

  neural_net->num_inputs = num_inputs;
  neural_net->num_outputs = num_outputs;
  neural_net->num_hidden_layers = num_hidden_layers;
  neural_net->neurons_per_hidden_layer = neurons_per_hidden_layer;
  neural_net->learning_rate = learning_rate;

  neural_net->layers = malloc((num_hidden_layers + 1) * sizeof(Layer));
  assert(neural_net->layers != NULL);

  int i = 0, j = 0, k = 0;

  if (num_hidden_layers > 0) {

    // create input layer
    neural_net->layers[0] = Layer_create(neurons_per_hidden_layer, num_inputs);

    // create hidden layers
    for (i = 1; i < num_hidden_layers; i++) {
      neural_net->layers[i] =
          Layer_create(neurons_per_hidden_layer, neurons_per_hidden_layer);
    }
    // create output layer
    neural_net->layers[num_hidden_layers] =
        Layer_create(num_outputs, neurons_per_hidden_layer);

  } else {
    // create single layer
    neural_net->layers[0] = Layer_create(num_outputs, num_inputs);
  }

  printf("\n\nNeural network parameters:\n");
  printf("\n\tnum inputs:\t\t\t%d\n", neural_net->num_inputs);
  printf("\tnum outputs:\t\t\t%d\n", neural_net->num_outputs);
  printf("\tnum hidden layers:\t\t%d\n", neural_net->num_hidden_layers);
  printf("\tneurons per hidden layer:\t%d\n",
         neural_net->neurons_per_hidden_layer);
  printf("\tlearning rate:\t\t\t%.1f\n", neural_net->learning_rate);

  return neural_net;
}

TrainingSet *TrainingSet_create(int num_inputs, double *inputs,
                                int num_outputs) {

  assert(inputs != NULL);

  TrainingSet *training_set = malloc(sizeof(TrainingSet));
  assert(training_set != NULL);

  training_set->inputs = malloc(num_inputs * sizeof(double));
  assert(training_set->inputs != NULL);

  training_set->desired_output = malloc(num_outputs * sizeof(double));
  assert(training_set->desired_output != NULL);

  for (int i = 0; i < num_inputs; i++) {
    training_set->inputs[i] = inputs[i];
  }

  return training_set;
}

void Update_weights(NeuralNet *neural_net, double *desired_output,
                    double *result) {

  assert(neural_net != NULL && desired_output != NULL && result != NULL);

  double error = 0;
  double error_gradient_sum = 0;

  int i = 0, j = 0, k = 0, p = 0;

  // loop through layers from last to first (backpropagation)
  for (i = neural_net->num_hidden_layers; i >= 0; i--) {

    // loop through neurons
    for (j = 0; j < neural_net->layers[i]->num_neurons; j++) {

      // if output layer, compute error
      if (i == neural_net->num_hidden_layers) {

        error = desired_output[j] - result[j];

        // error gradient calculated with delta rule
        neural_net->layers[i]->neurons[j]->error_gradient =
            result[j] * (1 - result[j]) * error;

      } else {
        neural_net->layers[i]->neurons[j]->error_gradient =
            neural_net->layers[i]->neurons[j]->output *
            (1 - neural_net->layers[i]->neurons[j]->output);

        error_gradient_sum = 0;

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
        if (i == neural_net->num_hidden_layers) {

          error = desired_output[j] - result[j];

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

double ArcTan(double value) { return atan(value); }

double BinaryStep(double value) { return value < 0 ? 0 : 1; }

double ELU(double value, double alpha) {
  return value <= 0 ? alpha * (exp(value) - 1) : value;
}

double LeakyReLU(double value, double alpha) {
  return value < 0 ? alpha * value : value;
}

double ReLU(double value) { return value > 0 ? value : 0; }

double Sigmoid(double value) { return 1 / (1 + exp(-value)); }

double Sinusoid(double value) { return sin(value); }

double TanH(double value) {
  return (exp(value) - exp(-value)) / (exp(value) + exp(-value));
}

double Activation_function(double value) { return Sigmoid(value); }

double Activation_function_output_layer(double value) { return Sigmoid(value); }

void Train(NeuralNet *neural_net, TrainingSet *training_set, double *result) {

  assert(neural_net != NULL && training_set != NULL && result != NULL);

  int i = 0, j = 0, k = 0;

  double training_inputs[neural_net->num_inputs];
  double N = 0;

  // copy inputs into temp array
  for (i = 0; i < neural_net->num_inputs; i++) {
    training_inputs[i] = training_set->inputs[i];
  }

  // loop through layers
  for (i = 0; i <= neural_net->num_hidden_layers; i++) {

    /*
    printf("\n\n  Layer %d:\n", i + 1);

    for (j = 0; j < neural_net->layers[i]->num_neurons; j++) {

      printf("\n     Neuron %d:\n", j + 1);
      printf("\n\tbias:\t\t\t%9f\n", neural_net->layers[i]->neurons[j]->bias);
      printf("\toutput:\t\t\t%9f\n", neural_net->layers[i]->neurons[j]->output);
      printf("\terror gradient:\t\t%9f\n",
             neural_net->layers[i]->neurons[j]->error_gradient);

      for (k = 0; k < neural_net->layers[i]->neurons[j]->num_inputs; k++) {
        printf("\tinput %d\t\t\t%9f\n", k + 1,
               neural_net->layers[i]->neurons[j]->inputs[k]);
      }
      for (k = 0; k < neural_net->layers[i]->neurons[j]->num_inputs; k++) {
        printf("\tweight %d:\t\t%9f\n", k + 1,
               neural_net->layers[i]->neurons[j]->weights[k]);
      }
    }
    */

    // if not input layer, set inputs to previous layer's outputs
    if (i > 0) {
      for (j = 0; j < neural_net->num_inputs; j++) {
        training_inputs[j] = result[0];
      }
    }

    // clear outputs
    for (j = 0; j < neural_net->num_outputs; j++) {
      result[j] = 0;
    }

    // loop through neurons
    for (j = 0; j < neural_net->layers[i]->num_neurons; j++) {

      N = 0;

      // clear neuron's inputs
      for (k = 0; k < neural_net->layers[i]->neurons[j]->num_inputs; k++) {
        neural_net->layers[i]->neurons[j]->inputs[k] = 0;
      }

      // loop through inputs
      for (k = 0; k < neural_net->layers[i]->neurons[j]->num_inputs; k++) {

        neural_net->layers[i]->neurons[j]->inputs[k] = training_inputs[k];

        // compute dot product
        N += neural_net->layers[i]->neurons[j]->weights[k] * training_inputs[k];
      }

      // subtract bias
      N -= neural_net->layers[i]->neurons[j]->bias;

      // compute output for output layer
      if (i == neural_net->num_hidden_layers) {
        neural_net->layers[i]->neurons[j]->output =
            Activation_function_output_layer(N);
      } else {
        // compute output for hidden layers
        neural_net->layers[i]->neurons[j]->output = Activation_function(N);
      }

      result[0] = neural_net->layers[i]->neurons[j]->output;
    }
  }

  Update_weights(neural_net, training_set->desired_output, result);
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

  for (int i = 0; i <= neural_net->num_hidden_layers; i++) {
    Layer_destroy(neural_net->layers[i]);
  }
  free(neural_net->layers);

  free(neural_net);
}

void TrainingSet_destroy(TrainingSet *training_set) {

  assert(training_set != NULL);

  free(training_set->inputs);
  free(training_set->desired_output);

  free(training_set);
}
