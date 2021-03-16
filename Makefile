CFLAGS = -std=c17 -Wall -Werror -Wextra -Wpedantic -g -O2

apps/neuralnet : src/main.c build/neuralnet.o
	gcc -o apps/neuralnet src/main.c build/neuralnet.o

build/neuralnet.o : libs/neuralnet.c libs/headers/neuralnet.h
	gcc -c -fpic -o build/neuralnet.o libs/neuralnet.c

clean : 
	rm apps/neuralnet build/neuralnet.o
