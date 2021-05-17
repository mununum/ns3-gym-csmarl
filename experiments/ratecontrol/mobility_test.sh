#!/bin/bash

# ndcf test

mkdir -p ndcf_mobility_result

for layout in node link
do
    for mobility in paired random
    do
        for graphSeed in {0..4}
        do
            python ns3_rollout.py --loss=geometric --topology=10,0.3 --episodes=5 \
                   --layout=$layout --mobility=$mobility --graphSeed=$graphSeed \
                   > ndcf_mobility_result/$layout-$mobility-$graphSeed.out &
        done
        wait
    done
done

# odcf test

rm -rf odcf_mobility_result
mkdir -p odcf_mobility_result

for simSeed in {1..5}
do
    for layout in node link
    do
        for mobility in paired random
        do
            for graphSeed in {0..4}
            do
                LD_LIBRARY_PATH=../../build/lib ../../build/scratch/csmarl_test/csmarl_test \
                --loss=geometric --topology=10,0.3 --layout=$layout --mobility=$mobility \
                --graphSeed=$graphSeed --simSeed=$simSeed >> odcf_mobility_result/$layout-$mobility-$graphSeed.out &
            done
        done
    done
    wait
done

