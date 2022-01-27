CFLAGS = -std=c17 -Wall -Werror -Wextra -Wpedantic -g -O2

bin/neuralnet : src/main.c obj/neuralnet.o
	gcc -o bin/neuralnet src/main.c obj/neuralnet.o -lm

obj/neuralnet.o : src/neuralnet.c lib/neuralnet.h
	gcc -c -fpic -o obj/neuralnet.o src/neuralnet.c

.PHONY: clean
clean : 
	rm bin/neuralnet obj/neuralnet.o
