# run executable with CPUPROFILE environment var set
LD_PRELOAD=/usr/lib/libprofiler.so CPUPROFILE=/tmp/prof.out ./bin/neuralnet

# run pprof and view results in web browser
pprof -http=localhost:8000 ./bin/neuralnet /tmp/prof.out

# run pprof to output one line per procedure 
#pprof --text ./bin/neuralnet /tmp/prof.out

# run pprof to get graphical output
#pprof --gv ./bin/neuralnet /tmp/prof.out

# generate PDF report with previous graphical output
#pprof --pdf ./bin/neuralnet /tmp/prof.out > /test/test_profile.pdf
