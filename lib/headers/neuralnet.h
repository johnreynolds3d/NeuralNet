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

extern struct Neuron *Neuron_create(int num_inputs);
extern struct Layer *Layer_create(int num_neurons, int num_neuron_inputs);

extern void Neuron_destroy(struct Neuron *neuron);
extern void Layer_destroy(struct Layer *layer);

#endif
