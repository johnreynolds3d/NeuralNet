CFLAGS = -std=c17 -Wall -Werror -Wextra -Wpedantic -g -O2

../../bin/mac_os/neuralnet : ../../src/main.c ../../build/mac_os/neuralnet.o
	gcc -o ../../bin/mac_os/neuralnet ../../src/main.c ../../build/mac_os/neuralnet.o

../../build/mac_os/neuralnet.o : ../../lib/neuralnet.c ../../lib/headers/neuralnet.h
	gcc -c -fpic -o ../../build/mac_os/neuralnet.o ../../lib/neuralnet.c

clean : 
	rm ../../bin/mac_os/neuralnet ../../build/mac_os/neuralnet.o 
