#include "../lib/headers/neuralnet.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {

  srand(time(NULL));

  // training data for logical XOR operation
  double inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  double desired_outputs[4][1] = {{0}, {1}, {1}, {0}};

  int num_inputs = sizeof(inputs[0]) / sizeof(inputs[0][0]);
  int num_outputs = sizeof(desired_outputs[0]) / sizeof(desired_outputs[0][0]);
  int num_hidden_layers = 1;
  int neurons_per_hidden_layer = 2;

  double alpha = 0.8;

  int num_training_sets = sizeof(inputs) / sizeof(inputs[0]);
  TrainingSet *training_sets[num_training_sets];

  printf("\n\n");

  int i;

  for (i = 0; i < num_training_sets; i++) {

    training_sets[i] = TrainingSet_create(inputs[i], desired_outputs[i]);

    printf("\ttraining_set[%d]->inputs[0]:\t\t%d\n", i,
           (int)training_sets[i]->inputs[0]);
    printf("\ttraining_set[%d]->inputs[1]:\t\t%d\n", i,
           (int)training_sets[i]->inputs[1]);
    printf("\ttraining_set[%d]->desired_outputs[0]:\t%d\n\n", i,
           (int)training_sets[i]->desired_outputs[0]);
  }

  NeuralNet *neural_net =
      NeuralNet_create(num_inputs, num_outputs, num_hidden_layers,
                       neurons_per_hidden_layer, alpha);
  assert(neural_net != NULL);

  printf("\n\tneural_net->num_inputs:\t\t\t%d\n", neural_net->num_inputs);
  printf("\tneural_net->num_outputs:\t\t%d\n", neural_net->num_outputs);
  printf("\tneural_net->num_hidden_layers:\t\t%d\n",
         neural_net->num_hidden_layers);
  printf("\tneural_net->neurons_per_hidden_layer:\t%d\n",
         neural_net->neurons_per_hidden_layer);
  printf("\tneural_net->alpha:\t\t\t%f\n\n", neural_net->alpha);

  for (i = 0; i < sizeof(neural_net->layers) / sizeof(neural_net->layers[0]);
       i++) {
    for (int j = 0; j < sizeof(neural_net->layers[0]->neurons) /
                            sizeof(neural_net->layers[0]->neurons[0]);
         j++) {
      printf("\tlayers[%d]->neurons[%d]->bias:\t\t%f\n", i, j,
             neural_net->layers[i]->neurons[j]->bias);
      printf("\tlayers[%d]->neurons[%d]->output:\t\t%f\n", i, j,
             neural_net->layers[i]->neurons[j]->output);
      printf("\tlayers[%d]->neurons[%d]->error_gradient:\t%f\n\n", i, j,
             neural_net->layers[i]->neurons[j]->error_gradient);
    }
  }

  /*
  double outputs[num_outputs];
  double sum_square_error;

  for (i = 0; i < 1; i++) {

    sum_square_error = 0;

    Train(neural_net, training_sets[0]);
    sum_square_error += pow(outputs[0] - 0, 2);

    printf("trying to train training_sets[%d]..\n", i);

    Train(neural_net, training_sets[1]);
    sum_square_error += pow(outputs[0] - 1, 2);

    Train(neural_net, training_sets[2]);
    sum_square_error += pow(outputs[0] - 1, 2);

    Train(neural_net, training_sets[3]);
    sum_square_error += pow(outputs[0] - 0, 2);
  }
  printf("\n\nsum_square_error:\t%f\n", sum_square_error);

  Train(neural_net, training_sets[0]);
  printf("[0 0] %f\n", outputs[0]);

  Train(neural_net, training_sets[1]);
  printf("[1 0] %f\n", outputs[0]);

  Train(neural_net, training_sets[2]);
  printf("[0 1] %f\n", outputs[0]);

  Train(neural_net, training_sets[3]);
  printf("[1 1] %f\n\n\n", outputs[0]);

  for (i = 0; i < num_training_sets; i++) {
    TrainingSet_destroy(training_sets[i]);
  }
  */

  NeuralNet_destroy(neural_net);

  return 0;
}
