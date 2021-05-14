# start cpu profiler
LD_PRELOAD=/usr/local/lib/libprofiler.so CPUPROFILE=test/cpu.prof ./bin/neuralnet

# run pprof to get graphical output
pprof -gv ./bin/neuralnet test/cpu.prof

# generate PDF report with previous graphical output
#pprof --pdf ./bin/neuralnet test/cpu.prof > test/cpu-profile.pdf
