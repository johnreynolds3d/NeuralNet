#ifndef neuralnet_h
#define neuralnet_h

typedef struct Neuron {
  int num_inputs;
  double bias;
  double output;
  double error_gradient;
  double *weights;
  double *inputs;
} Neuron;

typedef struct Layer {
  int num_neurons;
  Neuron **neurons;
} Layer;

typedef struct NeuralNet {
  int num_inputs;
  int num_outputs;
  int num_hidden_layers;
  int num_neurons_per_hidden_layer;
  double alpha;
  Layer **layers;
} NeuralNet;

Neuron *Neuron_create(int num_inputs);

Layer *Layer_create(int num_neurons, int num_neuron_inputs);

NeuralNet *NeuralNet_create(int num_inputs, int num_outputs,
                            int num_hidden_layers,
                            int num_neurons_per_hidden_layer, double alpha);

void Neuron_destroy(Neuron *neuron);

void Layer_destroy(Layer *layer);

void NeuralNet_destroy(NeuralNet *neural_net);

#endif
