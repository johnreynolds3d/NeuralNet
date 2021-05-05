#include "headers/neuralnet.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Neuron *Neuron_create(const int num_inputs) {

  Neuron *neuron = malloc(sizeof(*neuron));
  assert(neuron != NULL);

  neuron->bias = 0;
  neuron->output = 0;
  neuron->error_gradient = 0;
  neuron->num_inputs = num_inputs;

  neuron->inputs = malloc(num_inputs * sizeof(double));
  assert(neuron->inputs != NULL);

  neuron->weights = malloc(num_inputs * sizeof(double));
  assert(neuron->weights != NULL);

  for (int i = 0; i < num_inputs; i++) {
    neuron->inputs[i] = 0;
    neuron->weights[i] = (double)rand() / RAND_MAX * 2 - 1;
  }

  return neuron;
}

Layer *Layer_create(const int num_neurons, const int num_neuron_inputs) {

  Layer *layer = malloc(sizeof(*layer));
  assert(layer != NULL);

  layer->num_neurons = num_neurons;

  layer->neurons = malloc(num_neurons * sizeof(Neuron));
  assert(layer->neurons != NULL);

  for (int i = 0; i < num_neurons; i++) {
    layer->neurons[i] = Neuron_create(num_neuron_inputs);
  }

  return layer;
}

NeuralNet *NeuralNet_create(const int num_inputs, const int num_outputs,
                            const int num_hidden_layers,
                            const int neurons_per_hidden_layer,
                            const double learning_rate) {

  NeuralNet *neural_net = malloc(sizeof(*neural_net));
  assert(neural_net != NULL);

  neural_net->num_inputs = num_inputs;
  neural_net->num_outputs = num_outputs;
  neural_net->learning_rate = learning_rate;
  neural_net->num_hidden_layers = num_hidden_layers;
  neural_net->neurons_per_hidden_layer = neurons_per_hidden_layer;

  neural_net->layers = malloc((num_hidden_layers + 1) * sizeof(Layer));
  assert(neural_net->layers != NULL);

  if (num_hidden_layers > 0) {

    // create input layer
    neural_net->layers[0] = Layer_create(neurons_per_hidden_layer, num_inputs);

    // create hidden layers
    for (int i = 1; i < num_hidden_layers; i++) {
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

  return neural_net;
}

TrainingSet *TrainingSet_create(const double *inputs, const int num_inputs,
                                const double *desired_output) {

  assert(inputs != NULL && desired_output != NULL);

  TrainingSet *training_set = malloc(sizeof(*training_set));
  assert(training_set != NULL);

  training_set->act_func_hidden = 0;
  training_set->act_func_output = 0;
  training_set->desired_output = (double *)desired_output;

  training_set->inputs = malloc(num_inputs * sizeof(double));
  assert(training_set->inputs != NULL);

  for (int i = 0; i < num_inputs; i++) {
    training_set->inputs[i] = inputs[i];
  }

  training_set->results = calloc(1, sizeof(double));
  assert(training_set->results != NULL);

  return training_set;
}

void NeuralNet_print(const NeuralNet *neural_net) {

  assert(neural_net != NULL);

  int i, j, k;

  for (i = 0; i <= neural_net->num_hidden_layers; i++) {

    if (neural_net->num_hidden_layers == 0) {
      printf("\n\tSingle Layer:\n");
    } else {
      if (i == 0) {
        printf("\n\tInput Layer:\n");
      } else {
        if (i < neural_net->num_hidden_layers) {
          printf("\n\tHidden Layer %d:\n", i);
        } else {
          printf("\n\tOutput Layer:\n");
        }
      }
    }

    for (j = 0; j < neural_net->layers[i]->num_neurons; j++) {

      printf("\n\t  Neuron %d:\n\n", j + 1);

      printf("\t\tbias:\t\t %11.8f\n", neural_net->layers[i]->neurons[j]->bias);

      printf("\t\toutput:\t\t %11.8f\n",
             neural_net->layers[i]->neurons[j]->output);

      printf("\t\terror gradient:  %11.8f\n",
             neural_net->layers[i]->neurons[j]->error_gradient);

      for (k = 0; k < neural_net->layers[i]->neurons[j]->num_inputs; k++) {
        printf("\t\tinput %d:\t %11.8f\n", k + 1,
               neural_net->layers[i]->neurons[j]->inputs[k]);
      }

      for (k = 0; k < neural_net->layers[i]->neurons[j]->num_inputs; k++) {
        printf("\t\tweight %d:\t %11.8f\n", k + 1,
               neural_net->layers[i]->neurons[j]->weights[k]);
      }
    }
    printf("\n");
  }
  printf("\n");
}

void Update_weights(NeuralNet *neural_net, const double *desired_output,
                    const double *results) {

  assert(neural_net != NULL && desired_output != NULL && results != NULL);

  double error;
  double error_gradient_sum;

  int i, j, k, p;

  // loop through layers from last to first (backpropagation)
  for (i = neural_net->num_hidden_layers; i >= 0; i--) {

    // loop through neurons
    for (j = 0; j < neural_net->layers[i]->num_neurons; j++) {

      // if output layer, compute error
      if (i == neural_net->num_hidden_layers) {

        error = desired_output[j] - results[j];

        // error gradient calculated with delta rule
        neural_net->layers[i]->neurons[j]->error_gradient =
            results[j] * (1 - results[j]) * error;

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

          error = desired_output[j] - results[j];

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

double ArcTan(const double value) { return atan(value); }

double BinaryStep(const double value) { return value < 0 ? 0 : 1; }

double ELU(const double value, const double alpha) {
  return value <= 0 ? alpha * (exp(value) - 1) : value;
}

double LeakyReLU(const double value, const double alpha) {
  return value < 0 ? alpha * value : value;
}

double ReLU(const double value) { return value > 0 ? value : 0; }

double Sigmoid(const double value) { return 1 / (1 + exp(-value)); }

double Sinusoid(const double value) { return sin(value); }

double TanH(const double value) {
  return (exp(value) - exp(-value)) / (exp(value) + exp(-value));
}

double Act_func_hidden(const double value, const int function) {

  assert(function >= 0 && function <= 7);

  const double alpha = 0.01;

  switch (function) {

  case 0:
    return ArcTan(value);

  case 1:
    return BinaryStep(value);

  case 2:
    return ELU(value, alpha);

  case 3:
    return Sigmoid(value);

  case 4:
    return Sinusoid(value);

  case 5:
    return TanH(value);

  case 6:
    return LeakyReLU(value, alpha);

  case 7:
    return ReLU(value);
  }
}

double Act_func_output(const double value, const int function) {

  assert(function >= 0 && function <= 5);

  const double alpha = 0.01;

  switch (function) {

  case 0:
    return ArcTan(value);

  case 1:
    return BinaryStep(value);

  case 2:
    return ELU(value, alpha);

  case 3:
    return Sigmoid(value);

  case 4:
    return Sinusoid(value);

  case 5:
    return TanH(value);
  }
}

void Train(TrainingSet *training_set) {

  assert(training_set != NULL);

  NeuralNet *neural_net = training_set->neural_net;

  int i, j, k;

  double N;
  double training_inputs[neural_net->num_inputs];

  // copy inputs into temp array
  for (i = 0; i < neural_net->num_inputs; i++) {
    training_inputs[i] = training_set->inputs[i];
  }

  // loop through layers
  for (i = 0; i <= neural_net->num_hidden_layers; i++) {

    // if not input layer, set inputs to previous layer's outputs
    if (i > 0) {
      for (j = 0; j < neural_net->num_inputs; j++) {
        training_inputs[j] = training_set->results[0];
      }
    }

    // clear outputs
    for (j = 0; j < neural_net->num_outputs; j++) {
      training_set->results[j] = 0;
    }

    // loop through neurons
    for (j = 0; j < neural_net->layers[i]->num_neurons; j++) {

      // clear neuron's inputs
      for (k = 0; k < neural_net->layers[i]->neurons[j]->num_inputs; k++) {
        neural_net->layers[i]->neurons[j]->inputs[k] = 0;
      }

      N = 0;

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
            Act_func_output(N, training_set->act_func_output);

      } else {

        // compute output for hidden layers
        neural_net->layers[i]->neurons[j]->output =
            Act_func_hidden(N, training_set->act_func_hidden);
      }

      training_set->results[0] = neural_net->layers[i]->neurons[j]->output;
    }
  }

  Update_weights(neural_net, training_set->desired_output,
                 training_set->results);
}

void *PreTraining(void *arg) {

  assert(arg != NULL);

  const char *operations[] = {"AND", "NAND", "OR", "NOR", "XOR", "XNOR"};

  const char *activation_functions[] = {"ArcTan",     "Binary Step", "ELU",
                                        "Sigmoid",    "Sinusoid",    "TanH",
                                        "Leaky ReLU", "ReLU"};

  const double inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};

  const double outputs[6][4] = {{0, 0, 0, 1}, {1, 1, 1, 0}, {0, 1, 1, 1},
                                {1, 0, 0, 0}, {0, 1, 1, 0}, {1, 0, 0, 1}};

  const int i = *(int *)arg;
  const int num_outputs = 1;
  const int num_neural_nets = 50;
  const int num_epochs = pow(2, 9);
  const int neurons_per_hidden_layer = 2;
  const int num_inputs = sizeof(inputs[0]) / sizeof(inputs[0][0]);
  const int num_training_sets = sizeof(inputs) / sizeof(inputs[0]);
  const int num_activation_functions =
      sizeof(activation_functions) / sizeof(activation_functions[0]);

  double learning_rate;
  double sum_square_error;
  double best_results[num_training_sets];
  double best_sum_square_error = DBL_MAX;

  int best_neural_net;
  int num_hidden_layers;
  int best_act_func_hidden;
  int best_act_func_output;

  int j, k, n, p, q;

  NeuralNet *neural_nets[num_neural_nets];
  TrainingSet *training_sets[num_training_sets];

  // create training sets for current operation
  for (j = 0; j < num_training_sets; j++) {
    training_sets[j] =
        TrainingSet_create(inputs[j], num_inputs, &outputs[i][j]);
  }

  /*
   *  Create neural networks containing all possible combinations of hidden
   *  layers between 0 and 4, and learning rates between 0.0 and 1.0.
   */
  for (j = 0; j < 5; j++) {

    num_hidden_layers = j;

    for (k = 0; k < 10; k++) {

      learning_rate = 0.1 * k;

      neural_nets[j * 10 + k] =
          NeuralNet_create(num_inputs, num_outputs, num_hidden_layers,
                           neurons_per_hidden_layer, learning_rate);
    }
  }

  // loop through neural networks
  for (j = 0; j < num_neural_nets; j++) {

    for (k = 0; k < num_training_sets; k++) {
      training_sets[k]->neural_net = neural_nets[j];
    }

    // loop through activation functions for hidden layers
    for (k = 2; k < num_activation_functions; k++) {

      for (n = 0; n < num_training_sets; n++) {
        training_sets[n]->act_func_hidden = k;
      }

      // loop through activation functions for output layer (exclude ReLUs)
      for (n = 2; n < num_activation_functions - 2; n++) {

        for (p = 0; p < num_training_sets; p++) {
          training_sets[p]->act_func_output = n;
        }

        for (p = 0; p < num_epochs; p++) {

          sum_square_error = 0;

          for (q = 0; q < num_training_sets; q++) {

            Train(training_sets[q]);

            sum_square_error += pow(training_sets[q]->results[0] -
                                        training_sets[q]->desired_output[0],
                                    2);
          }
        }
        if (sum_square_error < best_sum_square_error) {

          best_neural_net = j;
          best_act_func_hidden = k;
          best_act_func_output = n;
          best_sum_square_error = sum_square_error;

          for (p = 0; p < num_training_sets; p++) {
            best_results[p] = training_sets[p]->results[0];
          }
        }
      }
    }
  }

  printf("\n     Best performance on %s was...\n\n", operations[i]);

  for (j = 0; j < num_training_sets; j++) {
    printf("\t\t[%d %d] %11.8f\n", (int)inputs[j][0], (int)inputs[j][1],
           best_results[j]);
  }

  printf("\n\t\tSSE:  %11.8f\n", best_sum_square_error);

  printf("\n        By Neural Network %d:\n\n", best_neural_net);

  printf("\t\thidden layers:    %d\n",
         neural_nets[best_neural_net]->num_hidden_layers);

  printf("\t\tlearning rate:    %.1f\n",
         neural_nets[best_neural_net]->learning_rate);

  printf("\t\tact func hidden:  %s\n",
         activation_functions[best_act_func_hidden]);

  printf("\t\tact func output:  %s\n",
         activation_functions[best_act_func_output]);

  for (j = 0; j < num_neural_nets; j++) {
    NeuralNet_destroy(neural_nets[j]);
  }

  for (j = 0; j < num_training_sets; j++) {
    TrainingSet_destroy(training_sets[j]);
  }

  return NULL;
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
  free(training_set->results);

  free(training_set);
}
