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

extern struct Neuron *Neuron_create(int num_inputs);

extern void Neuron_destroy(struct Neuron *neuron);

#endif
