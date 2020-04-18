for seed in `seq 1 10`
do
    LD_LIBRARY_PATH=./build/lib build/scratch/csmarl/csmarl --topology=complex --simSeed=$seed --simTime=60 --fixedFlow=true
done