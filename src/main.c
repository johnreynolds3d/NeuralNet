#include "../lib/headers/neuralnet.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {

  srand(time(NULL));

  // ----------------------------  OPERATIONS  ---------------------------------

  char *operations[] = {"AND", "NAND", "OR", "NOR", "XOR", "XNOR"};

  double inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};

  double outputs[6][4] = {{0, 0, 0, 1}, {1, 1, 1, 0}, {0, 1, 1, 1},
                          {1, 0, 0, 0}, {0, 1, 1, 0}, {1, 0, 0, 1}};

  int num_operations = sizeof(operations) / sizeof(operations[0]);

  // ------------------------  NETWORK PARAMETERS  -----------------------------

  int num_inputs = 2;
  int num_outputs = 1;
  int num_hidden_layers = 1;
  int neurons_per_hidden_layer = 2;

  double learning_rate = 0.8;

  printf("\n\nNeural network parameters:\n\n");
  printf("\t\tnum inputs:\t\t   %d\n", num_inputs);
  printf("\t\tnum outputs:\t\t   %d\n", num_outputs);
  printf("\t\tnum hidden layers:\t   %d\n", num_hidden_layers);
  printf("\t\tneurons per hidden layer:  %d\n", neurons_per_hidden_layer);
  printf("\t\tlearning rate:\t\t   %.1f\n", learning_rate);

  // -----------------------  ACTIVATION FUNCTIONS  ----------------------------

  char *activation_functions[] = {"ArcTan",     "Binary Step", "ELU",
                                  "Leaky ReLU", "ReLU",        "Sigmoid",
                                  "Sinusoid",   "TanH"};

  int num_activation_functions =
      sizeof(activation_functions) / sizeof(activation_functions[0]);

  int best_activation_function = 0;

  // -----------------------------  TRAINING  ----------------------------------

  int num_training_sets = sizeof(outputs[0]) / sizeof(outputs[0][0]);
  TrainingSet *training_sets[num_training_sets];

  int i = 0, j = 0, k = 0, p = 0;

  for (i = 0; i < num_training_sets; i++) {
    training_sets[i] = TrainingSet_create(num_inputs, inputs[i], num_outputs);
  }

  double result[num_outputs];
  double sum_square_error = 0;
  double best_sum_square_error;

  int num_epochs = pow(2, 11);

  // loop through operations
  for (i = 0; i < num_operations; i++) {

    printf("\n\n  Training on %s for %d epochs...\n\n", operations[i],
           num_epochs);

    // reinitialize training sets
    for (j = 0; j < num_training_sets; j++) {

      training_sets[j]->desired_output[0] = outputs[i][j];

      printf("\t\t[%d %d]  %d\n", (int)training_sets[j]->inputs[0],
             (int)training_sets[j]->inputs[1],
             (int)training_sets[j]->desired_output[0]);
    }

    best_sum_square_error = DBL_MAX; // reset best sse

    // loop throgh activation functions
    for (j = 0; j < num_activation_functions; j++) {

      // create new network
      NeuralNet *neural_net =
          NeuralNet_create(num_inputs, num_outputs, num_hidden_layers,
                           neurons_per_hidden_layer, learning_rate);

      for (k = 0; k < num_epochs; k++) {

        sum_square_error = 0; // reset sse

        for (p = 0; p < num_training_sets; p++) {

          Train(neural_net, training_sets[p], result, j);

          sum_square_error +=
              pow(result[0] - training_sets[p]->desired_output[0], 2);
        }
      }

      // final training and printing of results
      printf("\n\n    Results for %s using %s:\n\n", operations[i],
             activation_functions[j]);

      for (k = 0; k < num_training_sets; k++) {

        Train(neural_net, training_sets[k], result, j);

        printf("\t\t[%d %d] %9f\n", (int)training_sets[k]->inputs[0],
               (int)training_sets[k]->inputs[1], result[0]);
      }
      printf("\n\t\tSSE:  %9f\n", sum_square_error);

      // keep track of best activation function and SSE
      if (sum_square_error < best_sum_square_error) {
        best_sum_square_error = sum_square_error;
        best_activation_function = j;
      }

      // TODO save network values
      // printf("\n  Network values:\n");
      // NeuralNet_print(neural_net);

      NeuralNet_destroy(neural_net);
    }

    if (best_sum_square_error != DBL_MAX) {
      printf("\n\n\tThe best performing activation function for %s\n\twas %s, "
             "with a sum square error of%9f\n\n",
             operations[i], activation_functions[best_activation_function],
             best_sum_square_error);
    }
  }
  printf("\n\n");

  for (j = 0; j < num_training_sets; j++) {
    TrainingSet_destroy(training_sets[j]);
  }

  return 0;
}
