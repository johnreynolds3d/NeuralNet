# neuralnet

### A Basic Implementation of an Artificial Neural Network in C

This project uses an Artificial Neural Network to learn and perform the logical operations AND, NAND, OR, NOR, XOR, and XNOR. 

![Screenshot](/img/neuralnet.webp?raw=true "")

A work in progress, my goal is to enable this system to discover the optimal machine learning models for these tasks by generating, and evaluating the performance of, various combinations of network parameters and activation functions.

## Installation

  1. Clone this repository
     ```
     # git clone git@github.com:johnreynolds3d/neuralnet.git
     ```
  2. cd into the source directory
     ```
     # cd neuralnet 
     ```
  3. Run make
     ```
     # make
     ```
  4. Then run the program with
     ```
     # ./bin/neuralnet
     ```

## Testing

I've been using Valgrind to highlight potential memory issues. 

To test using Valgrind, run
```
# sh test/runtests.sh
```

## Credits

This project is based on Penny de Byl's *A Beginner's Guide To Machine Learning with Unity* https://www.udemy.com/course/machine-learning-with-unity/
