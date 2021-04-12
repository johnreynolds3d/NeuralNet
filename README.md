# neuralnet

### A basic implementation of an Artificial Neural Network in C

Let's see if we can't round up a whole BUNCH of those pesky Perceptrons, and get 'em to learn the logical XOR operation! (and a few other ones, while we're at it...)

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

## Tests

I've been using Valgrind to highlight potential memory issues. 

On Linux, run
```
# sh test/runtests.sh
```

## Credits

This project is based on Penny de Byl's fabulous https://www.udemy.com/course/machine-learning-with-unity/

## Built with

  * Pop!\_OS 20.10
  * Vim 8.2
  * Valgrind 3.18.0
  * gcc 10.2.0
