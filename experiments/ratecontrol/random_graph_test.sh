#!/bin/bash

if [ $# -lt 1 ]; then
    echo "usage:" $0 "[batch_size]"
    exit 1
fi
batch_size=$1

count=0
mkdir -p random_graph_result

for N in `seq 20 -2 2`
do
    for d in `seq 1 -0.1 0.1`
    do
        for i in `seq 0 4`
        do
            python ns3_rollout.py --loss=geometric --topology=$N,$d --graphSeed=$i --episodes=5 > random_graph_result/geometric-$N-$d-$i.out &
            count=$((count+1))

            if [ $count -ge $batch_size ]
            then
                wait
                count=0
            fi
        done
    done
done

wait