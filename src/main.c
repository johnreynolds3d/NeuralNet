#include "../lib/headers/neuralnet.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

int main() {

  srand(time(NULL));

  // ----------------------------  OPERATIONS  ---------------------------------

  char *operations[6] = {"AND", "NAND", "OR", "NOR", "XOR", "XNOR"};

  double inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};

  double outputs[6][4] = {{0, 0, 0, 1}, {1, 1, 1, 0}, {0, 1, 1, 1},
                          {1, 0, 0, 0}, {0, 1, 1, 0}, {1, 0, 0, 1}};

  int num_operations = sizeof(operations) / sizeof(operations[0]);

  // -----------------------  ACTIVATION FUNCTIONS  ----------------------------

  char *activation_functions[8] = {"ArcTan",     "Binary Step", "ELU",
                                   "Sigmoid",    "Sinusoid",    "TanH",
                                   "Leaky ReLU", "ReLU"};

  int best_act_funcs_hidden[num_operations];
  int best_act_funcs_output[num_operations];
  int num_activation_functions =
      sizeof(activation_functions) / sizeof(activation_functions[0]);

  // --------------------------  NEURAL NETWORKS  ------------------------------

  int num_outputs = 1;
  int num_neural_nets = 50;
  int num_hidden_layers;
  int neurons_per_hidden_layer = 2;
  int best_neural_nets[num_operations];
  int num_inputs = sizeof(inputs[0]) / sizeof(inputs[0][0]);

  double learning_rate;

  NeuralNet *neural_nets[num_neural_nets];

  // ------------------------------  THREADS  ----------------------------------

  int num_epochs = pow(2, 12);
  int num_training_sets = sizeof(inputs) / sizeof(inputs[0]);

  pthread_t threads[num_training_sets];

  TrainingSet *training_sets[num_training_sets];

  int i = 0, j = 0, k = 0, n = 0, p = 0, q = 0;

  for (i = 0; i < num_training_sets; i++) {
    training_sets[i] = TrainingSet_create(inputs[i], num_inputs, num_epochs);
  }

  // ------------------------------  TRAINING  ---------------------------------

  int result_code = 0;
  double sum_square_error;
  double best_sum_square_errors[num_operations];
  double best_results[num_operations][num_training_sets];

  // loop through operations
  for (i = 0; i < num_operations; i++) {

    best_sum_square_errors[i] = DBL_MAX;

    printf("\n\n  Training on %s:\n\n", operations[i]);

    // reinitialize and print training sets for current operation
    for (j = 0; j < num_training_sets; j++) {

      training_sets[j]->desired_output[0] = outputs[i][j];

      printf("\t\t[%d %d]  %d\n", (int)training_sets[j]->inputs[0],
             (int)training_sets[j]->inputs[1],
             (int)training_sets[j]->desired_output[0]);
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
      for (k = 3; k < num_activation_functions; k++) {

        for (n = 0; n < num_training_sets; n++) {
          training_sets[n]->act_func_hidden = k;
        }

        // loop through activation functions for output layer (exclude ReLUs)
        for (n = 3; n < num_activation_functions - 2; n++) {

          sum_square_error = 0;

          for (p = 0; p < num_training_sets; p++) {
            training_sets[p]->act_func_output = n;
          }

          for (p = 0; p < num_training_sets; p++) {

            result_code =
                pthread_create(&threads[p], NULL, Train, training_sets[p]);
            assert(!result_code);

            // TrainingSet_print(training_sets[p]);

            sum_square_error += pow(training_sets[p]->results[0] -
                                        training_sets[p]->desired_output[0],
                                    2);
          }
          for (p = 0; p < num_training_sets; p++) {
            result_code = pthread_join(threads[p], NULL);
            assert(!result_code);
          }

          if (sum_square_error < best_sum_square_errors[i]) {

            best_neural_nets[i] = j;
            best_act_funcs_hidden[i] = k;
            best_act_funcs_output[i] = n;
            best_sum_square_errors[i] = sum_square_error;

            for (p = 0; p < num_training_sets; p++) {
              best_results[i][p] = training_sets[p]->results[0];
            }
          }
        }
      }
    }

    // -----------------------------  RESULTS  ---------------------------------

    printf("\n     Best performance on %s was...\n\n", operations[i]);

    for (j = 0; j < num_training_sets; j++) {
      printf("\t\t[%d %d] %35.32f\n", (int)inputs[j][0], (int)inputs[j][1],
             best_results[i][j]);
    }

    printf("\n\t\tSSE:  %35.32f\n", best_sum_square_errors[i]);

    printf("\n        By Neural Network %d:\n\n", best_neural_nets[i]);

    printf("\t\thidden layers:    %d\n",
           neural_nets[best_neural_nets[i]]->num_hidden_layers);

    printf("\t\tlearning rate:    %.1f\n",
           neural_nets[best_neural_nets[i]]->learning_rate);

    printf("\t\tact func hidden:  %s\n",
           activation_functions[best_act_funcs_hidden[i]]);

    printf("\t\tact func output:  %s\n",
           activation_functions[best_act_funcs_output[i]]);

    for (j = 0; j < num_neural_nets; j++) {
      NeuralNet_destroy(neural_nets[j]);
    }
  }
  printf("\n\n");

  for (i = 0; i < num_training_sets; i++) {
    TrainingSet_destroy(training_sets[i]);
  }

  return 0;
}
