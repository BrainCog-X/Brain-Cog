# Developmental Plasticity-inspired Adaptive Pruning for Deep Spiking and Artificial Neural Networks #

## Requirments ##
* numpy
* timm
* pytorch >= 1.7.0
* collections
* argparse

## Introduction ##
Dynamic Structure Development of Spiking Neural Networks (DSD-SNN) for efficient and adaptive continual learning: 
the DSD-SNN dynamically grows new neurons and prunes redundant neurons, increasing memory capacity and reducing computational overhead.
the DSD-SNN verlap shared structure to leverage acquired knowledge to new tasks, empowering a single network capable of supporting multiple incremental tasks. 
We validate the effectiveness of the DSD-SNN multiple class incremental learning and task incremental learning benchmarks.

## Run ##
cd ./cifar100
```CUDA_VISIBLE_DEVICES=0 python main_simplified.py```
cd ./mnist
```CUDA_VISIBLE_DEVICES=0 python main_simplified.py```

Enjoy!
