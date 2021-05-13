#valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes -s ./bin/neuralnet

# start profiling
LD_PRELOAD=/usr/local/lib/libprofiler.so CPUPROFILE=test/test.prof ./bin/neuralnet

# get a graphical output
#pprof -gv ./bin/neuralnet test/test.prof

# display graphical output in a web browser
#pprof --web ./bin/neuralnet test/test.prof

# generate PDF report with previous graphical output
pprof --pdf ./bin/neuralnet test/test.prof > test/output.pdf
