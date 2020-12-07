#!/bin/bash

for N in `seq 20 -2 2`
do
    for d in `seq 1 -0.1 0.6`
    do
        for i in `seq 0 4`
        do
            python ns3_rollout.py --topology=complex_graphs/complex-$N-$d-$i --episodes=1 > random_graph_result/complex-$N-$d-$i.out &
        done
    done
    wait

    for d in `seq 0.5 -0.1 0.1`
    do
        for i in `seq 0 4`
        do
            python ns3_rollout.py --topology=complex_graphs/complex-$N-$d-$i --episodes=1 > random_graph_result/complex-$N-$d-$i.out &
        done
    done
    wait
done