#include "../lib/headers/neuralnet.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

char *operations[] = {"AND", "NAND", "OR", "NOR", "XOR", "XNOR"};

double inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};

double outputs[6][4] = {{0, 0, 0, 1}, {1, 1, 1, 0}, {0, 1, 1, 1},
                        {1, 0, 0, 0}, {0, 1, 1, 0}, {1, 0, 0, 1}};

int num_operations = sizeof(operations) / sizeof(operations[0]);

char *activation_functions[] = {"ArcTan",     "Binary Step", "ELU",
                                "Sigmoid",    "Sinusoid",    "TanH",
                                "Leaky ReLU", "ReLU"};

int num_activation_functions =
    sizeof(activation_functions) / sizeof(activation_functions[0]);

int num_training_sets = sizeof(inputs) / sizeof(inputs[0]);

void *PreTraining(void *arg) {

  assert(arg != NULL);

  int num_outputs = 1;
  int num_hidden_layers;
  int num_neural_nets = 50;
  int neurons_per_hidden_layer = 2;
  int num_inputs = sizeof(inputs[0]) / sizeof(inputs[0][0]);

  double learning_rate;
  double best_results[num_training_sets];

  int best_neural_net;
  int best_act_func_hidden;
  int best_act_func_output;
  int num_epochs = pow(2, 10);

  int i = *(int *)arg;
  int j = 0, k = 0, n = 0, p = 0, q = 0;

  TrainingSet *training_sets[num_training_sets];

  // printf("\n\n  Training on %s:\n\n", operations[i]);

  // create and print training sets for current operation
  for (j = 0; j < num_training_sets; j++) {

    training_sets[j] =
        TrainingSet_create(inputs[j], num_inputs, &outputs[i][j]);

    /*
    printf("\t\t[%d %d]  %d\n", (int)training_sets[j]->inputs[0],
           (int)training_sets[j]->inputs[1],
           (int)training_sets[j]->desired_output[0]);
           */
  }

  NeuralNet *neural_nets[num_neural_nets];

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

  double sum_square_error;
  double best_sum_square_error = DBL_MAX;

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

  // -----------------------------  RESULTS  ---------------------------------

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

int main() {

  srand(time(NULL));

  int result_code;

  pthread_t threads[num_operations];

  int i = 0;

  // loop through operations
  for (i = 0; i < num_operations; i++) {

    result_code = pthread_create(&threads[i], NULL, PreTraining, &i);
    assert(!result_code);
  }
  for (i = 0; i < num_operations; i++) {
    result_code = pthread_join(threads[i], NULL);
    assert(!result_code);
  }

  printf("\n\n");

  return 0;
}
