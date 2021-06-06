# run executable with CPUPROFILE environment var set
LD_PRELOAD=/usr/lib/libprofiler.so CPUPROFILE=test/test.prof ./bin/neuralnet

# run pprof and view results in web browser
pprof -http=localhost:8000 ./bin/neuralnet test/test.prof

# run pprof to output one line per procedure 
#pprof --text ./bin/neuralnet test/test.prof

# run pprof to get graphical output
#pprof -gv ./bin/neuralnet test/test.prof

# generate PDF report with previous graphical output
pprof --pdf ./bin/neuralnet test/test.prof > test/test_profile.pdf
