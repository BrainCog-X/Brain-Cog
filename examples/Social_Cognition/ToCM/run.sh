#!/bin/sh
seed_max=10
#for seed in `seq ${seed_max}`;
#do
#    echo "seed is ${seed}:"
#    python train.py
#    kill Main_Thread
#done

#seed_max=10  # 设置最大的种子值，这里假设为10

#for./run, seed in $(seq 1 $seed_max); do
#    echo "seed is $seed:"
#    python train.py --seed $seed  # 将当前种子值作为参数传递给 train.py
#done
python train.py --seed 50
pkill Main_Thread
python train.py --seed 50
pkill Main_Thread
#python train.py --seed 1
#pkill Main_Thread