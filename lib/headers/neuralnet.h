#ifndef neuralnet_h
#define neuralnet_h

struct Neuron {
  int num_inputs;
  double bias;
  double output;
  double error_gradient;
  double *weights;
  double *inputs;
};

struct Layer {
  int num_neurons;
  struct Neuron **neurons;
};

struct NeuralNet {
  int num_inputs;
  int num_outputs;
  int num_hidden_layers;
  int num_neurons_per_hidden_layer;
  double alpha;
  struct Layer **layers;
};

struct Neuron *Neuron_create(int num_inputs);

struct Layer *Layer_create(int num_neurons, int num_neuron_inputs);

struct NeuralNet *NeuralNet_create(int num_inputs, int num_outputs,
                                   int num_hidden_layers,
                                   int num_neurons_per_hidden_layer,
                                   double alpha);

void Neuron_destroy(struct Neuron *neuron);

void Layer_destroy(struct Layer *layer);

void NeuralNet_destroy(struct NeuralNet *neural_net);

#endif
