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

  int num_neural_nets = 4;

  NeuralNet *neural_nets[num_neural_nets];

  int num_inputs = 2;
  int num_outputs = 1;
  int num_hidden_layers;
  int neurons_per_hidden_layer = 2;

  double learning_rate;

  int best_neural_net;
  int worst_neural_net;

  // -----------------------  ACTIVATION FUNCTIONS  ----------------------------

  int num_activation_functions = 8;

  char *activation_functions[] = {"ArcTan",     "Binary Step", "ELU",
                                  "Leaky ReLU", "ReLU",        "Sigmoid",
                                  "Sinusoid",   "TanH"};

  int best_activation_function;
  int worst_activation_function;

  // -----------------------------  TRAINING  ----------------------------------

  int num_training_sets = 4;

  TrainingSet *training_sets[num_training_sets];

  int i = 0, j = 0, k = 0, p = 0, q = 0;

  for (i = 0; i < num_training_sets; i++) {
    training_sets[i] = TrainingSet_create(num_inputs, inputs[i], num_outputs);
  }

  double sum_square_error = 0;
  double best_sum_square_error;
  double worst_sum_square_error;
  double result[num_outputs];

  int num_epochs = pow(2, 11);

  printf("\n\n\n");

  // loop through operations
  for (i = 0; i < num_operations; i++) {

    printf("Training for %d epochs on %s:\n\n", num_epochs, operations[i]);

    // reinitialize training sets for current operation
    for (j = 0; j < num_training_sets; j++) {

      training_sets[j]->desired_output[0] = outputs[i][j];

      printf("\t\t[%d %d]  %d\n", (int)training_sets[j]->inputs[0],
             (int)training_sets[j]->inputs[1],
             (int)training_sets[j]->desired_output[0]);
    }

    best_sum_square_error = DBL_MAX;
    worst_sum_square_error = 0;

    // create neural networks with random hyperparameters
    for (j = 0; j < num_neural_nets; j++) {

      num_hidden_layers = rand() % (4 - 0 + 1) + 0;
      learning_rate = (double)rand() / (double)RAND_MAX;

      neural_nets[j] =
          NeuralNet_create(num_inputs, num_outputs, num_hidden_layers,
                           neurons_per_hidden_layer, learning_rate);
    }

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

        /*
        printf("\n\n    Results for %s using Neural Network %d w/ %s:\n\n ",
               operations[i], j + 1, activation_functions[k]);
               */

        // final training and printing of results
        for (p = 0; p < num_training_sets; p++) {

          Train(neural_nets[j], training_sets[p], result, k);

          /*
          printf("\t\t[%d %d] %9f\n", (int)training_sets[p]->inputs[0],
                 (int)training_sets[p]->inputs[1], result[0]);
                 */
        }
        // printf("\n\t\tSSE:  %9f\n", sum_square_error);

        if (sum_square_error < best_sum_square_error) {
          best_sum_square_error = sum_square_error;
          best_activation_function = k;
          best_neural_net = j;
        }

        if (sum_square_error > worst_sum_square_error) {
          worst_sum_square_error = sum_square_error;
          worst_activation_function = k;
          worst_neural_net = j;
        }
      }
    }

    // -----------------------------  TESTING  ---------------------------------

    printf("\n\n  Our BEST performer on %s, with an SSE of%9f was...\n\n"
           "        Neural Network %d:\n\n",
           operations[i], best_sum_square_error, best_neural_net + 1);

    printf("\t\tnum hidden layers:    %d\n",
           neural_nets[best_neural_net]->num_hidden_layers);

    printf("\t\tlearning rate:        %.1f\n",
           neural_nets[best_neural_net]->learning_rate);

    printf("\t\tactivation function:  %s\n\n",
           activation_functions[best_activation_function]);

    result[0] = 0;

    for (j = 0; j < num_epochs; j++) {

      sum_square_error = 0;

      for (k = 0; k < num_training_sets; k++) {

        Train(neural_nets[best_neural_net], training_sets[k], result,
              best_activation_function);

        sum_square_error +=
            pow(result[0] - training_sets[k]->desired_output[0], 2);
      }
    }

    printf("\n        Testing our BEST performer on %s produces:\n\n",
           operations[i]);

    for (j = 0; j < num_training_sets; j++) {

      Train(neural_nets[best_neural_net], training_sets[j], result,
            best_activation_function);

      printf("\t\t[%d %d] %9f\n", (int)training_sets[j]->inputs[0],
             (int)training_sets[j]->inputs[1], result[0]);
    }
    printf("\n\t\tSSE:  %9f\n\n", sum_square_error);

    // NeuralNet_print(neural_nets[best_neural_net]);

    /*
        printf("\n  Our WORST performer on %s, with an SSE of%9f was...\n\n"
               "        Neural Network %d:\n\n",
               operations[i], worst_sum_square_error, worst_neural_net + 1);

        printf("\t\tnum hidden layers:    %d\n",
               neural_nets[worst_neural_net]->num_hidden_layers);

        printf("\t\tlearning rate:        %.1f\n",
               neural_nets[worst_neural_net]->learning_rate);

        printf("\t\tactivation function:  %s\n\n",
               activation_functions[worst_activation_function]);

        result[0] = 0;

        for (j = 0; j < num_epochs; j++) {

          sum_square_error = 0;

          for (k = 0; k < num_training_sets; k++) {

            Train(neural_nets[worst_neural_net], training_sets[k], result,
                  worst_activation_function);

            sum_square_error +=
                pow(result[0] - training_sets[k]->desired_output[0], 2);
          }
        }

        printf("\n      Testing our WORST performer on %s produces:\n\n",
               operations[i]);

        for (j = 0; j < num_training_sets; j++) {

          Train(neural_nets[worst_neural_net], training_sets[j], result,
                worst_activation_function);

          printf("\t\t[%d %d] %9f\n", (int)training_sets[j]->inputs[0],
                 (int)training_sets[j]->inputs[1], result[0]);
        }
        printf("\n\t\tSSE:  %9f\n\n", sum_square_error);
        */

    // NeuralNet_print(neural_nets[worst_neural_net]);

    printf("\n\n");

    for (j = 0; j < num_neural_nets; j++) {
      NeuralNet_destroy(neural_nets[j]);
    }
  }

  for (i = 0; i < num_training_sets; i++) {
    TrainingSet_destroy(training_sets[i]);
  }

  return 0;
}
