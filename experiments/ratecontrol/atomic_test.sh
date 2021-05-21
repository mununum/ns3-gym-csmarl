#!/bin/bash

# ndcf test

mkdir -p ndcf_atomic_result

for topology in ht ia ch2 ch3 ch4
do
    python ns3_rollout.py --layout=node --loss=graph --episodes=5 --topology=$topology \
    > ndcf_atomic_result/$topology.out &
done
wait

# odcf test

rm -rf odcf_atomic_result
mkdir -p odcf_atomic_result

pwd=$PWD
cd ../..

for simSeed in {1..5}
do
    for topology in ht ia ch2 ch3 ch4
    do
        LD_LIBRARY_PATH=build/lib build/scratch/csmarl_test/csmarl_test \
        --layout=node --loss=graph --topology=$topology --debug=true --algorithm=odcf \
        >> $pwd/odcf_atomic_result/$topology.out &
    done
    wait
done

cd -

# 80211 test

rm -rf 80211_atomic_result
mkdir -p 80211_atomic_result

cd ../..

for simSeed in {1..5}
do
    for topology in ht ia ch2 ch3 ch4
    do
        LD_LIBRARY_PATH=build/lib build/scratch/csmarl_test/csmarl_test \
        --layout=node --loss=graph --topology=$topology --debug=true --algorithm=80211 \
        >> $pwd/80211_atomic_result/$topology.out &
    done
    wait
done

cd -