#  Sequence Production SNN
[![DOI](https://doi.org/10.3389/fncom.2021.612041)]

This repository contains code from our paper [**Brain Inspired Sequences Production by Spiking Neural Networks With Reward-Modulated STDP**] published in Frontiers in Computational Neuroscience. https://www.frontiersin.org/articles/10.3389/fncom.2021.612041/full. If you use our code or refer to this project, please cite this paper.

## Requirments

* numpy
* scipy
* pytorch >= 1.7.0
* torchvision





## Run

```shell
python main.py file
```


This module builds a sequence Production spiking neural network model, realizing the memory and reconstruction functions for arbitrary symbol sequences. The input causal graph is shown in figure causal_graph.png. The input current, spike trains during the learning process and the network weight distribution after the learning are shown in the Results folder.
