#include "../lib/headers/neuralnet.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {

  srand(time(NULL));

  int num_operations = 6;
  int num_sets = 4;

  char *operations[] = {"AND", "NAND", "OR", "NOR", "XOR", "XNOR"};

  double inputs[4][2] = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};

  double outputs[6][4] = {{0.0, 0.0, 0.0, 1.0}, {1.0, 1.0, 1.0, 0.0},
                          {0.0, 1.0, 1.0, 1.0}, {1.0, 0.0, 0.0, 0.0},
                          {0.0, 1.0, 1.0, 0.0}, {1.0, 0.0, 0.0, 1.0}};

  int num_inputs = 2;
  int num_outputs = 1;
  int num_hidden_layers = 1;
  int neurons_per_hidden_layer = 2;

  double learning_rate = 0.8;

  NeuralNet *neural_net =
      NeuralNet_create(num_inputs, num_outputs, num_hidden_layers,
                       neurons_per_hidden_layer, learning_rate);

  TrainingSet *training_sets[num_sets];

  double sum_square_error = 0.0;

  int epochs = pow(2, 2);

  int i = 0, j = 0, k = 0;

  for (i = 0; i < 1; i++) {

    int random_operation = rand() % ((num_operations - 1) - 0 + 1) + 0;

    printf("\n\n Training for %d epochs on %s:\n", epochs,
           operations[random_operation]);

    for (j = 0; j < num_sets; j++) {
      printf("\n\t[%d %d]  %d", (int)inputs[j][0], (int)inputs[j][1],
             (int)outputs[random_operation][j]);
    }

    for (j = 0; j < num_sets; j++) {
      training_sets[j] = TrainingSet_create(num_inputs, inputs[j], num_outputs,
                                            &outputs[random_operation][j]);
    }

    double result[num_outputs];
    sum_square_error = 0.0;

    for (j = 0; j < epochs; j++) {

      printf("\n\n\n Epoch %d:", j);

      sum_square_error = 0.0;

      for (k = 0; k < num_sets; k++) {
        Train(neural_net, training_sets[k], result);
        sum_square_error +=
            pow(result[0] - training_sets[k]->desired_output[0], 2);
      }
    }
    printf("\n    Sum square error:\t%.9f\n", sum_square_error);

    // final training and printing of results
    for (j = 0; j < num_sets; j++) {
      Train(neural_net, training_sets[j], result);
    }

    printf("\n\n  Results:\n\n");

    for (j = 0; j < num_sets; j++) {
      printf("\t[%d %d] %f\n", (int)training_sets[j]->inputs[0],
             (int)training_sets[j]->inputs[1], result[0]);
    }

    for (j = 0; j < num_sets; j++) {
      TrainingSet_destroy(training_sets[j]);
    }
  }
  printf("\n\n");

  NeuralNet_destroy(neural_net);

  return 0;
}
