# run executable with HEAPPROFILE environment var set
HEAPPROFILE=test/heapprof HEAPCHECK=draconian ./bin/neuralnet

# run pprof to output one line per procedure 
#pprof --text ./bin/neuralnet test/heapprof.0001.heap

# run pprof to get graphical output
#pprof -gv ./bin/neuralnet test/heapprof.0001.heap

# generate PDF report with previous graphical output
pprof --pdf ./bin/neuralnet test/heapprof.0001.heap > test/heapprof.0001.heap.pdf
