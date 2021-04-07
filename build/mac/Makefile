CFLAGS = -std=c17 -Wall -Werror -Wextra -Wpedantic -g -O2

../../bin/mac/neuralnet : ../../src/main.c ../../build/mac/neuralnet.o
	gcc -o ../../bin/mac/neuralnet ../../src/main.c ../../build/mac/neuralnet.o

../../build/mac/neuralnet.o : ../../lib/neuralnet.c ../../lib/headers/neuralnet.h
	gcc -c -fpic -o ../../build/mac/neuralnet.o ../../lib/neuralnet.c

clean : 
	rm ../../bin/mac/neuralnet ../../build/mac/neuralnet.o 
