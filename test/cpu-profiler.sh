# run executable with CPUPROFILE environment var set
LD_PRELOAD=/usr/local/lib/libprofiler.so CPUPROFILE=test/cpu.prof ./bin/neuralnet

# run pprof to output one line per procedure 
pprof --text ./bin/neuralnet test/cpu.prof

# run pprof to get graphical output
#pprof -gv ./bin/neuralnet test/cpu.prof

# generate PDF report with previous graphical output
#pprof --pdf ./bin/neuralnet test/cpu.prof > test/cpu-profile.pdf
