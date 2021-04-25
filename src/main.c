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

  // -------------------------  NEURAL NETWORKS  -------------------------------

  int num_inputs = 2;
  int num_outputs = 1;
  int num_hidden_layers = 1;
  int neurons_per_hidden_layer = 2;

  double learning_rate = 0.8;

  int num_neural_nets = 50;

  NeuralNet *neural_nets[num_neural_nets];

  int best_neural_net_id[num_operations];
  NeuralNet *best_neural_nets[num_operations];

  // -----------------------  ACTIVATION FUNCTIONS  ----------------------------

  int num_activation_functions = 8;

  char *activation_functions[] = {"ArcTan",     "Binary Step", "ELU",
                                  "Sigmoid",    "Sinusoid",    "TanH",
                                  "Leaky ReLU", "ReLU"};

  int best_act_funcs_hidden[num_operations];
  int best_act_funcs_output[num_operations];

  // -----------------------------  TRAINING  ----------------------------------

  int num_training_sets = 4;

  TrainingSet *training_sets[num_training_sets];

  int i = 0, j = 0, k = 0, n = 0, p = 0, q = 0;

  for (i = 0; i < num_training_sets; i++) {
    training_sets[i] = TrainingSet_create(num_inputs, inputs[i], num_outputs);
  }

  double results[num_outputs];
  double sum_square_error = 0;
  double prev_results[6][4] = {0};
  double best_results[6][4] = {0};
  double best_sum_square_errors[num_operations];

  int num_epochs = pow(2, 13);

  // loop through operations
  for (i = 0; i < num_operations; i++) {

    results[0] = 0;
    best_neural_net_id[i] = 0;
    best_act_funcs_hidden[i] = 0;
    best_act_funcs_output[i] = 0;
    best_sum_square_errors[i] = DBL_MAX;

    printf("\n\n  Training on %s:\n\n", operations[i]);

    // reinitialize training sets for current operation
    for (j = 0; j < num_training_sets; j++) {

      training_sets[j]->desired_output[0] = outputs[i][j];

      printf("\t\t[%d %d]  %d\n", (int)training_sets[j]->inputs[0],
             (int)training_sets[j]->inputs[1],
             (int)training_sets[j]->desired_output[0]);
    }

    /*
     * Create neural networks containing all possible combinations of hidden
     * layers between 0 and 4, and learning rates between 0.0 and 1.0.
     */
    for (j = 0; j < 5; j++) {

      num_hidden_layers = j;

      for (k = 0; k < 10; k++) {

        learning_rate = (double)k / 10;

        neural_nets[j * 10 + k] =
            NeuralNet_create(num_inputs, num_outputs, num_hidden_layers,
                             neurons_per_hidden_layer, learning_rate);
      }
    }

    // loop through neural networks
    for (j = 0; j < num_neural_nets; j++) {

      // loop through activation functions for hidden layers
      for (k = 3; k < num_activation_functions; k++) {

        // loop through activation functions for output layer (exclude ReLUs)
        for (n = 3; n < num_activation_functions - 2; n++) {

          results[0] = 0;

          for (p = 0; p < num_epochs; p++) {

            sum_square_error = 0;

            for (q = 0; q < num_training_sets; q++) {

              Train(neural_nets[j], training_sets[q], results, k, n);

              sum_square_error +=
                  pow(results[0] - training_sets[q]->desired_output[0], 2);
            }
          }

          // final training and saving of results
          for (q = 0; q < num_training_sets; q++) {

            Train(neural_nets[j], training_sets[q], results, k, n);

            sum_square_error +=
                pow(results[0] - training_sets[q]->desired_output[0], 2);

            prev_results[i][q] = results[0];
          }

          if (sum_square_error < best_sum_square_errors[i]) {

            best_sum_square_errors[i] = sum_square_error;
            best_neural_nets[i] = neural_nets[j];
            best_neural_net_id[i] = j;
            best_act_funcs_hidden[i] = k;
            best_act_funcs_output[i] = n;

            for (q = 0; q < num_training_sets; q++) {
              best_results[i][q] = prev_results[i][q];
            }
          }
        }
      }
    }

    // -----------------------------  RESULTS  ---------------------------------

    printf("\n\n     Best performance on %s was...\n\n", operations[i]);

    for (j = 0; j < num_training_sets; j++) {
      printf("\t\t[%d %d] %35.32f\n", (int)inputs[j][0], (int)inputs[j][1],
             best_results[i][j]);
    }

    printf("\n\t\tSSE:  %35.32f\n", best_sum_square_errors[i]);

    printf("\n\n        By Neural Network %d:\n\n", best_neural_net_id[i]);

    printf("\t\thidden layers:    %d\n",
           best_neural_nets[i]->num_hidden_layers);

    printf("\t\tlearning rate:    %.1f\n", best_neural_nets[i]->learning_rate);

    printf("\t\tact func hidden:  %s\n",
           activation_functions[best_act_funcs_hidden[i]]);

    printf("\t\tact func output:  %s\n\n",
           activation_functions[best_act_funcs_output[i]]);

    // NeuralNet_print(best_neural_nets[i]);

    for (j = 0; j < num_neural_nets; j++) {
      NeuralNet_destroy(neural_nets[j]);
    }
  }
  printf("\n");

  for (i = 0; i < num_training_sets; i++) {
    TrainingSet_destroy(training_sets[i]);
  }

  return 0;
}
