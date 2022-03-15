CC = gcc
CFLAGS = -std=c17 -Wall -Werror -Wextra -Wpedantic -g -O2

bin/neuralnet : src/main.c obj/neuralnet.o
	@mkdir -p bin
	$(CC) $(CFLAGS) -o bin/neuralnet src/main.c obj/neuralnet.o -lm

obj/neuralnet.o : src/neuralnet.c lib/neuralnet.h
	@mkdir -p obj
	$(CC) $(CFLAGS) -c -fpic -o obj/neuralnet.o src/neuralnet.c

.PHONY: clean
clean : 
	rm bin/neuralnet obj/neuralnet.o
