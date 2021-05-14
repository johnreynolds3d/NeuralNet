# run executable with HEAPPROFILE environment var set
LD_PRELOAD=/usr/local/lib/libprofiler.so HEAPPROFILE=test/heapprof ./bin/neuralnet
#HEAPPROFILE=test/heapprof ./bin/neuralnet

# run pprof to analyse heap usage
pprof ./bin/neuralnet test/heapprof.0001.heap

# run pprof to get graphical output
pprof --gv ./bin/neuralnet test/heapprof.0001.heap

# generate PDF report with previous graphical output
#pprof --pdf ./bin/neuralnet test/heapprof.0045.heap > test/heapprof.0045.heap.pdf
