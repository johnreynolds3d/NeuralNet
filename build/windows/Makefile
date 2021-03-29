CFLAGS = -std=c17 -Wall -Werror -Wextra -Wpedantic -g -O2

../../bin/windows/neuralnet.exe : ../../src/main.c ../../build/windows/neuralnet.o
	x86_64-w64-mingw32-gcc -o ../../bin/windows/neuralnet.exe ../../src/main.c ../../build/windows/neuralnet.o

../../build/windows/neuralnet.o : ../../lib/neuralnet.c ../../lib/headers/neuralnet.h
	x86_64-w64-mingw32-gcc -c -fpic -o ../../build/windows/neuralnet.o ../../lib/neuralnet.c

clean : 
	rm ../../bin/windows/neuralnet_win64.exe ../../build/windows/neuralnet.o
