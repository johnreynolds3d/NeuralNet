#include "../lib/headers/neuralnet.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {

  srand(time(NULL));

  /*
  printf("\n\n  Training for logical AND operation\n");
  double inputs[4][2] = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
  double desired_outputs[4][1] = {{0.0}, {0.0}, {0.0}, {1.0}};
  */

  printf("\n\n  Training for logical NAND operation\n");
  double inputs[4][2] = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
  double desired_outputs[4][1] = {{1.0}, {1.0}, {1.0}, {0.0}};

  /*
  printf("\n\n  Training for logical OR operation\n");
  double inputs[4][2] = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
  double desired_outputs[4][1] = {{0.0}, {1.0}, {1.0}, {1.0}};

  printf("\n\n  Training for logical NOR operation\n");
  double inputs[4][2] = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
  double desired_outputs[4][1] = {{1.0}, {0.0}, {0.0}, {0.0}};

  printf("\n\n  Training for logical XOR operation\n");
  double inputs[4][2] = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
  double desired_outputs[4][1] = {{0.0}, {1.0}, {1.0}, {0.0}};

  printf("\n\n  Training for logical XNOR operation\n");
  double inputs[4][2] = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
  double desired_outputs[4][1] = {{1.0}, {0.0}, {0.0}, {1.0}};
  */

  int num_inputs = sizeof(inputs[0]) / sizeof(inputs[0][0]);
  int num_outputs = sizeof(desired_outputs[0]) / sizeof(desired_outputs[0][0]);
  int num_hidden_layers = 1;
  int neurons_per_hidden_layer = 2;

  double learning_rate = 0.8;

  NeuralNet *neural_net =
      NeuralNet_create(num_inputs, num_outputs, num_hidden_layers,
                       neurons_per_hidden_layer, learning_rate);

  int num_training_sets = sizeof(inputs) / sizeof(inputs[0]);

  TrainingSet *training_sets[num_training_sets];

  int i = 0, j = 0;

  for (i = 0; i < num_training_sets; i++) {
    training_sets[i] = TrainingSet_create(num_inputs, inputs[i], num_outputs,
                                          desired_outputs[i]);
  }

  double result[num_outputs];
  double sum_square_error = 0.0;

  int epochs = (int)pow(2, 10);

  printf("\n    Training epochs:\t\t\t\t %d\n", epochs);

  for (i = 0; i < epochs; i++) {

    sum_square_error = 0.0;

    for (j = 0; j < num_training_sets; j++) {
      Train(neural_net, training_sets[j], result);
      sum_square_error +=
          pow(result[0] - training_sets[j]->desired_output[0], 2);
    }
  }
  printf("\n\tsum_square_error:\t\t\t%9f\n\n", sum_square_error);

  // final training and printing of result
  printf("\n    Results:\n\n");

  for (j = 0; j < num_training_sets; j++) {
    Train(neural_net, training_sets[j], result);
    printf("\t[%d %d] %f\n", (int)training_sets[j]->inputs[0],
           (int)training_sets[j]->inputs[1], result[0]);
  }
  printf("\n\n");

  NeuralNet_destroy(neural_net);

  for (i = 0; i < num_training_sets; i++) {
    TrainingSet_destroy(training_sets[i]);
  }

  return 0;
}
