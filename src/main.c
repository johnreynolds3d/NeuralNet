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

  int num_operations = 6;

  char *operations[] = {"AND", "NAND", "OR", "NOR", "XOR", "XNOR"};

  double inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};

  double outputs[6][4] = {{0, 0, 0, 1}, {1, 1, 1, 0}, {0, 1, 1, 1},
                          {1, 0, 0, 0}, {0, 1, 1, 0}, {1, 0, 0, 1}};

  // ---------------------  NEURAL NETWORK PARAMETERS  -------------------------

  int num_inputs = 2;
  int num_outputs = 1;
  int num_hidden_layers = 1;
  int neurons_per_hidden_layer = 2;

  double learning_rate = 0.8;

  int num_neural_nets = 40;

  NeuralNet *neural_nets[num_neural_nets];

  int i = 0, j = 0, k = 0, p = 0, q = 0;

  for (i = 0; i < 4; i++) {

    num_hidden_layers = i;

    for (j = 0; j < 10; j++) {

      learning_rate = (double)j / 10;

      neural_nets[i * 10 + j] =
          NeuralNet_create(num_inputs, num_outputs, num_hidden_layers,
                           neurons_per_hidden_layer, learning_rate);
    }
  }

  NeuralNet *best_neural_nets[num_operations];

  // -----------------------  ACTIVATION FUNCTIONS  ----------------------------

  int num_activation_functions = 8;

  char *activation_functions[] = {"ArcTan",     "Binary Step", "ELU",
                                  "Leaky ReLU", "ReLU",        "Sigmoid",
                                  "Sinusoid",   "TanH"};

  int best_activation_functions[num_operations];

  // -----------------------------  TRAINING  ----------------------------------

  int num_training_sets = 4;

  TrainingSet *training_sets[num_training_sets];

  for (i = 0; i < num_training_sets; i++) {
    training_sets[i] = TrainingSet_create(num_inputs, inputs[i], num_outputs);
  }

  double sum_square_error = 0;
  double best_sum_square_errors[num_operations];
  double result[num_outputs];

  result[0] = 0;

  int num_epochs = pow(2, 10);

  printf("\n\n\n");

  // loop through operations
  for (i = 3; i < num_operations - 2; i++) {

    printf("\nTraining for %d epochs on %s:\n\n", num_epochs, operations[i]);

    // reinitialize training sets for current operation
    for (j = 0; j < num_training_sets; j++) {

      training_sets[j]->desired_output[0] = outputs[i][j];

      printf("\t\t[%d %d]  %d\n", (int)training_sets[j]->inputs[0],
             (int)training_sets[j]->inputs[1],
             (int)training_sets[j]->desired_output[0]);
    }

    best_sum_square_errors[i] = DBL_MAX;

    // loop through neural networks
    for (j = 0; j < num_neural_nets; j++) {

      // loop through activation functions
      for (k = 0; k < num_activation_functions; k++) {

        result[0] = 0;

        for (p = 0; p < num_epochs; p++) {

          sum_square_error = 0;

          for (q = 0; q < num_training_sets; q++) {

            Train(neural_nets[j], training_sets[q], result, k);

            sum_square_error +=
                pow(result[0] - training_sets[q]->desired_output[0], 2);
          }
        }

        // select the best performers found so far
        if (sum_square_error < best_sum_square_errors[i]) {

          printf("\n\n    Results for %s using Neural Network %d w/ %s:\n\n ",
                 operations[i], j + 1, activation_functions[k]);

          // final training and printing of results
          for (p = 0; p < num_training_sets; p++) {

            Train(neural_nets[j], training_sets[p], result, k);

            printf("\t\t[%d %d] %35.32f\n", (int)training_sets[p]->inputs[0],
                   (int)training_sets[p]->inputs[1], result[0]);
          }
          printf("\n\t\tSSE:  %35.32f\n\n", sum_square_error);

          // final check and saving of results
          if (sum_square_error < best_sum_square_errors[i]) {
            best_sum_square_errors[i] = sum_square_error;
            best_activation_functions[i] = k;
            best_neural_nets[i] = neural_nets[j];
          }
        }
      }
    }
  }

  // -----------------------------  RESULTS  ---------------------------------

  for (i = 3; i < num_operations - 2; i++) {

    printf("\n\n  Our best performer on %s was...\n\n", operations[i]);

    printf("\t\tnum hidden layers:\t%d\n",
           best_neural_nets[i]->num_hidden_layers);

    printf("\t\tlearning rate:\t\t%.1f\n", best_neural_nets[i]->learning_rate);

    printf("\t\tactivation function:\t%s\n",
           activation_functions[best_activation_functions[i]]);

    printf("\n\t\tSSE:\t%35.32f\n", best_sum_square_errors[i]);

    NeuralNet_print(best_neural_nets[i]);
  }

  printf("\n\n");

  for (i = 0; i < num_neural_nets; i++) {
    NeuralNet_destroy(neural_nets[i]);
  }

  for (i = 0; i < num_training_sets; i++) {
    TrainingSet_destroy(training_sets[i]);
  }

  return 0;
}
