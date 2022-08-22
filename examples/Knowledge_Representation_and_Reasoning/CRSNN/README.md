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


### Citation 
If you find this package helpful, please consider citing the following papers:

```BibTex
@inproceedings{fang2021CRSNN,
  title={A Brain-Inspired Causal Reasoning Model Based on Spiking Neural Networks},
  author={Fang, Hongjian and Zeng, Yi},
  booktitle={2021 International Joint Conference on Neural Networks (IJCNN)},
  pages={1--5},
  year={2021},
  organization={IEEE}
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
