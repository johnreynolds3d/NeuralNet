CFLAGS = -std=c17 -Wall -Werror -Wextra -Wpedantic -g -O2

bin/neuralnet : src/main.c build/neuralnet.o
	gcc -o bin/neuralnet src/main.c build/neuralnet.o -lm -lpthread -lprofiler -ltcmalloc

build/neuralnet.o : lib/neuralnet.c lib/headers/neuralnet.h
	gcc -c -fpic -o build/neuralnet.o lib/neuralnet.c

clean : 
	rm bin/neuralnet build/neuralnet.o tmp/heapprof.0001.heap tmp/prof.out test/test_profile.pdf test/heapprof.0001.heap.pdf test/results.html
