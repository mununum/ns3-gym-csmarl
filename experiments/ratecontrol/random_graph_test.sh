#!/bin/bash

count=0

for N in `seq 20 -2 2`
do
    for d in `seq 1 -0.1 0.1`
    do
        for i in `seq 0 4`
        do
            python ns3_rollout.py --topology=complex_graphs/complex-$N-$d-$i --episodes=5 > random_graph_result_complex_single/complex-$N-$d-$i.out &
            count=$((count+1))

            if [ $count -ge 16 ]
            then
                wait
                count=0
            fi
        done
    done
done

wait