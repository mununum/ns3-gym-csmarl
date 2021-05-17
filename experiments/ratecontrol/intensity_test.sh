#!/bin/bash

# ndcf test

mkdir -p ndcf_intensity_test

for graphSeed in {0..4}
do
    python ns3_rollout.py --loss=geometric --topology=10,0.3 --graphSeed=$graphSeed --intensity \
    > ndcf_intensity_test/$graphSeed.out &
done
wait

# odcf test

rm -rf odcf_intensity_test
mkdir -p odcf_intensity_test

for intensity in $(seq 0.1 0.1 1)
do
    for graphSeed in {0..4}
    do
        LD_LIBRARY_PATH=../../build/lib ../../build/scratch/csmarl_test/csmarl_test \
        --loss=geometric --topology=10,0.3 --graphSeed=$graphSeed --intensity=$intensity \
        >> odcf_intensity_test/$graphSeed.out &
    done
    wait
done