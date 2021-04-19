# neuralnet

### A basic implementation of an Artificial Neural Network in C

This project uses an Artificial Neural Network to learn and perform the logical operations AND, NAND, OR, NOR, XOR, and XNOR. 

A work in progress, my goal is to enable the program to discover the optimum machine learning models for these tasks by generating and evaluating the performance of different network parameters and activation functions.

## Installation

  1. Clone this repository
     ```
     # git clone git@github.com:johnreynolds3d/neuralnet.git
     ```
  2. cd into the source directory
     ```
     # cd neuralnet 
     ```
  3. Run make (on Linux; not sure about Windows atm...)
     ```
     # make -C build/linux
     ```
     or, if you're on a Mac
     ```
     # make -C build/mac
     ```
  4. Then run the program (on Linux) with
     ```
     # ./bin/linux/neuralnet
     ```
     on Mac
     ```
     # ./bin/mac/neuralnet
     ```

## Testing

I've been using Valgrind to highlight potential memory issues. 

On Linux, run
```
# sh test/runtests.sh
```

## Credits

This project is based on Penny de Byl's *A Beginner's Guide To Machine Learning with Unity* https://www.udemy.com/course/machine-learning-with-unity/
