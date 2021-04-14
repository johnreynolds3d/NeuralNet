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
  double inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  double desired_outputs[4][1] = {{0}, {0}, {0}, {1}};

  printf("\n\n  Training for logical NAND operation\n");
  double inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  double desired_outputs[4][1] = {{1}, {1}, {1}, {0}};
  */

  printf("\n\n  Training for logical OR operation\n");
  double inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  double desired_outputs[4][1] = {{0}, {1}, {1}, {1}};

  /*
  printf("\n\n  Training for logical NOR operation\n");
  double inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  double desired_outputs[4][1] = {{1}, {0}, {0}, {0}};

  printf("\n\n  Training for logical XOR operation\n");
  double inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  double desired_outputs[4][1] = {{0}, {1}, {1}, {0}};

  printf("\n\n  Training for logical XNOR operation\n");
  double inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  double desired_outputs[4][1] = {{1}, {0}, {0}, {1}};
  */

  int num_inputs = sizeof(inputs[0]) / sizeof(inputs[0][0]);
  int num_outputs = sizeof(desired_outputs[0]) / sizeof(desired_outputs[0][0]);
  int num_hidden_layers = 1;
  int neurons_per_hidden_layer = 2;

  double learning_rate = 0.8;

  NeuralNet *neural_net =
      NeuralNet_create(num_inputs, num_outputs, num_hidden_layers,
                       neurons_per_hidden_layer, learning_rate);

  printf("\n\tneural_net->num_inputs:\t\t\t% d\n", neural_net->num_inputs);

  printf("\tneural_net->num_outputs:\t\t% d\n", neural_net->num_outputs);

  printf("\tneural_net->num_hidden_layers:\t\t% d\n",
         neural_net->num_hidden_layers);

  printf("\tneural_net->neurons_per_hidden_layer:\t% d\n",
         neural_net->neurons_per_hidden_layer);

  printf("\tneural_net->learning_rate:\t\t%9f\n\n", neural_net->learning_rate);

  int num_layers =
      neural_net->num_hidden_layers > 0 ? neural_net->num_hidden_layers + 2 : 1;

  int i, j, k;

  /*
for (i = 0; i < num_layers; i++) {

for (j = 0; j < neural_net->layers[i]->num_neurons; j++) {

printf("\n\tlayers[%d]->neurons[%d]->bias:\t\t%9f\n", i, j,
       neural_net->layers[i]->neurons[j]->bias);

printf("\tlayers[%d]->neurons[%d]->output:\t\t%9f\n", i, j,
       neural_net->layers[i]->neurons[j]->output);

printf("\tlayers[%d]->neurons[%d]->error_gradient:\t%9f\n", i, j,
       neural_net->layers[i]->neurons[j]->error_gradient);

for (k = 0; k < neural_net->layers[i]->neurons[j]->num_inputs; k++) {
  printf("\tlayers[%d]->neurons[%d]->weights[%d]:\t%9f\n", i, j, k,
         neural_net->layers[i]->neurons[j]->weights[k]);
}

for (k = 0; k < neural_net->layers[i]->neurons[j]->num_inputs; k++) {
  printf("\tlayers[%d]->neurons[%d]->inputs[%d]:\t%9f\n", i, j, k,
         neural_net->layers[i]->neurons[j]->inputs[k]);
}
}
}
  */

  int num_training_sets = sizeof(inputs) / sizeof(inputs[0]);

  TrainingSet *training_sets[num_training_sets];

  for (i = 0; i < num_training_sets; i++) {

    training_sets[i] = TrainingSet_create(num_inputs, inputs[i], num_outputs,
                                          desired_outputs[i]);

    /*
printf("\n\ttraining_set[%d]->inputs[0]:\t\t% d\n", i,
(int)training_sets[i]->inputs[0]);

printf("\ttraining_set[%d]->inputs[1]:\t\t% d\n", i,
(int)training_sets[i]->inputs[1]);

printf("\ttraining_set[%d]->desired_outputs[0]:\t% d\n", i,
(int)training_sets[i]->desired_outputs[0]);
                             */
  }

  double outputs[num_outputs];
  double sum_square_error;

  int epochs = pow(2, 8);

  printf("\n    Training epochs:\t\t\t\t %d\n", epochs);

  for (i = 0; i < epochs; i++) {

    sum_square_error = 0;

    Train(neural_net, training_sets[0], outputs);
    sum_square_error += pow(outputs[0] - 0, 2); // (actual - desired output)^2

    Train(neural_net, training_sets[1], outputs);
    sum_square_error += pow(outputs[0] - 1, 2);

    Train(neural_net, training_sets[2], outputs);
    sum_square_error += pow(outputs[0] - 1, 2);

    Train(neural_net, training_sets[3], outputs);
    sum_square_error += pow(outputs[0] - 0, 2);
  }
  printf("\n\tsum_square_error:\t\t\t%9f\n\n", sum_square_error);

  // final training and printing of results
  printf("\n    Results:\n");

  Train(neural_net, training_sets[0], outputs);
  printf("\n\t[0 0] %f\n", outputs[0]);

  Train(neural_net, training_sets[1], outputs);
  printf("\t[1 0] %f\n", outputs[0]);

  Train(neural_net, training_sets[2], outputs);
  printf("\t[0 1] %f\n", outputs[0]);

  Train(neural_net, training_sets[3], outputs);
  printf("\t[1 1] %f\n\n\n", outputs[0]);

  NeuralNet_destroy(neural_net);

  for (i = 0; i < num_training_sets; i++) {
    TrainingSet_destroy(training_sets[i]);
  }

  return 0;
}
