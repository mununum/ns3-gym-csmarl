#!/bin/bash

rm -rf odcf_random_graph_result
mkdir -p odcf_random_graph_result

for seed in `seq 1 5`
do
    for N in `seq 20 -2 2`
    do
        for d in `seq 1 -0.1 0.6`
        do
            for i in `seq 0 4`
            do
                LD_LIBRARY_PATH=../../build/lib ../../build/scratch/csmarl_test/csmarl_test --loss=geometric --topology=$N,$d --graphSeed=$i --simSeed=$seed --debug=true --algorithm=odcf >> odcf_random_graph_result/complex-$N-$d-$i.out &
            done
        done
        wait
        for d in `seq 0.5 -0.1 0.1`
        do
            for i in `seq 0 4`
            do
                LD_LIBRARY_PATH=../../build/lib ../../build/scratch/csmarl_test/csmarl_test --loss=geometric --topology=$N,$d --graphSeed=$i --simSeed=$seed --debug=true --algorithm=odcf >> odcf_random_graph_result/complex-$N-$d-$i.out &
            done
        done
        wait
    done
done