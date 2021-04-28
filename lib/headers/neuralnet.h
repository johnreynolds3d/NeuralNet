#ifndef neuralnet_h
#define neuralnet_h

typedef struct Neuron {
  int num_inputs;
  double bias;
  double output;
  double error_gradient;
  double *inputs;
  double *weights;
} Neuron;

typedef struct Layer {
  int num_neurons;
  Neuron **neurons;
} Layer;

typedef struct NeuralNet {
  int num_inputs;
  int num_outputs;
  int num_hidden_layers;
  int neurons_per_hidden_layer;
  double learning_rate;
  Layer **layers;
} NeuralNet;

typedef struct TrainingSet {
  double *inputs;
  double *desired_output;
} TrainingSet;

typedef struct TrainingData {
  NeuralNet *neural_net;
  TrainingSet *training_set;
  double *results;
  int act_func_hidden;
  int act_func_output;
  int num_epochs;
} TrainingData;

Neuron *Neuron_create(int num_inputs);

Layer *Layer_create(int num_neurons, int num_neuron_inputs);

NeuralNet *NeuralNet_create(int num_inputs, int num_outputs,
                            int num_hidden_layers, int neurons_per_hidden_layer,
                            double learning_rate);

TrainingSet *TrainingSet_create(int num_inputs, double *inputs,
                                int num_outputs);

TrainingData *TrainingData_create(NeuralNet *neural_net,
                                  TrainingSet *training_set, double *results,
                                  int act_func_hidden, int act_func_output,
                                  int num_epochs);

void NeuralNet_print(NeuralNet *neural_net);

void Update_weights(NeuralNet *neural_net, double *desired_output,
                    double *result);

double ArcTan(double value);

double BinaryStep(double value);

double ELU(double value, double alpha);

double LeakyReLU(double value, double alpha);

double ReLU(double value);

double Sigmoid(double value);

double Sinusoid(double value);

double TanH(double value);

double Act_func_hidden(double value, int function);

double Act_func_output(double value, int function);

void *Train(void *arg);

void Neuron_destroy(Neuron *neuron);

void Layer_destroy(Layer *layer);

void NeuralNet_destroy(NeuralNet *neural_net);

void TrainingSet_destroy(TrainingSet *training_set);

#endif
