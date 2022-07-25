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


### Citation 
If you find this package helpful, please consider citing the following papers:

```BibTex
@article{fang2021spsnn,
    title     = {Brain inspired sequences production by spiking neural networks with reward-modulated stdp},
    author    = {Fang, Hongjian and Zeng, Yi and Zhao, Feifei},
    journal   = {Frontiers in Computational Neuroscience},
    volume    = {15},
    pages     = {8},
    year      = {2021},
    publisher = {Frontiers}
}


@misc{https://doi.org/10.48550/arxiv.2207.08533,
  doi = {10.48550/ARXIV.2207.08533},
  url = {https://arxiv.org/abs/2207.08533},
  author = {Zeng, Yi and Zhao, Dongcheng and Zhao, Feifei and Shen, Guobin and Dong, Yiting and Lu, Enmeng and Zhang, Qian and Sun, Yinqian and Liang, Qian and Zhao, Yuxuan and Zhao, Zhuoya and Fang, Hongjian and Wang, Yuwei and Li, Yang and Liu, Xin and Du, Chengcheng and Kong, Qingqun and Ruan, Zizhe and Bi, Weida},
  title = {BrainCog: A Spiking Neural Network based Brain-inspired Cognitive Intelligence Engine for Brain-inspired AI and Brain Simulation},
  publisher = {arXiv},
  year = {2022},
}

```
