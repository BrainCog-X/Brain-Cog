# Commonsense Knowledge Representation SNN

(https://arxiv.org/abs/2207.05561)

This repository contains code from our paper [**Brain-inspired Graph Spiking Neural Networks for Commonsense Knowledge Representation and Reasoning**] preprint in: https://arxiv.org/abs/2207.05561 . If you use our code or refer to this project, please cite this paper.




## Requirments

* python=3.8
* numpy
* scipy
* turicreate
* pytorch >= 1.7.0
* torchvision


## Dataset

ConceptNet: https://github.com/commonsense/conceptnet5 


## Run

```shell
python main.py
```

This module selects core knowledge in ConceptNet to form the sub_Concept.csv file as the input of the model. The input current, spike trains during the learning process and the network weight distribution after the learning are shown in the Results folder.
