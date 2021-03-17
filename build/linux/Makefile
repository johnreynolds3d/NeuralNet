CFLAGS = -std=c17 -Wall -Werror -Wextra -Wpedantic -g -O2

../../bin/linux/neuralnet : ../../src/main.c ../../build/linux/neuralnet.o
	gcc -o ../../bin/linux/neuralnet ../../src/main.c ../../build/linux/neuralnet.o

../../build/linux/neuralnet.o : ../../lib/neuralnet.c ../../lib/headers/neuralnet.h
	gcc -c -fpic -o ../../build/linux/neuralnet.o ../../lib/neuralnet.c

clean : 
	rm ../../bin/linux/neuralnet ../../build/linux/neuralnet.o
