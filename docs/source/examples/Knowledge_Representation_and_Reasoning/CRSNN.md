# Causal Reasoning SNN
(https://10.1109/IJCNN52387.2021.9534102)

This repository contains code from our paper [**A Brain-Inspired Causal Reasoning Model Based on Spiking Neural Networks
**] published in 2021 International Joint Conference on Neural Networks (IJCNN).  https://ieeexplore.ieee.org/abstract/document/9534102. If you use our code or refer to this project, please cite this paper.

## Requirments

* numpy
* scipy
* pytorch >= 1.7.0
* torchvision



## Run

```shell
python main.py
```


This module builds an example of a brain-like causal inference spiking neural network model. The input causal graph is shown in figure causal_graph.png. The input current, spike trains during the learning process and the network weight distribution after the learning are shown in the Results folder.
