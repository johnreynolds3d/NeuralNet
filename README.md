# neuralnet

**Gaze in disbelief as an Artificial Neural Network (AKA *Multilayer Perceptron*) does its darnedest to learn and perform the logical operations AND, NAND, OR, NOR, XOR, and XNOR!**

![Screenshot](/img/neuralnet.webp?raw=true "")

A work in progress, my goal is to enable this system to discover the optimal machine learning models for these tasks by generating (and evaluating the performance of) various combinations of network parameters and activation functions.

## Installation

  1. Clone this repository
     ```shell
     $ git clone git@github.com:johnreynolds3d/neuralnet.git
     ```
  2. `cd` into the source directory
     ```shell
     $ cd neuralnet 
     ```
  3. Run `make`
     ```shell
     $ make
     ```
  4. Run the program 
     ```shell
     $ ./bin/neuralnet
     ```

## Testing

I've been using [Valgrind](https://valgrind.org/) to highlight potential memory issues.

To test with Valgrind:
  ```shell
  $ ./memtest.sh
  ```
NOTE: run `chmod +x memtest.sh` if you don't have permission to execute the file.

## Credits

This project is based on Penny de Byl's fabulous tutorial [A Beginner's Guide To Machine Learning with Unity](https://www.udemy.com/course/machine-learning-with-unity/).
