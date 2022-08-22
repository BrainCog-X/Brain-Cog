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


### Citation 
If you find this package helpful, please consider citing the following papers:

```BibTex
@article{KRRfang2022,
    title   = {Brain-inspired Graph Spiking Neural Networks for Commonsense Knowledge Representation and Reasoning},
    author  = { Fang, Hongjian and Zeng, Yi and  Tang, Jianbo and Wang, Yuwei and Liang, Yao and  Liu, Xin},
    journal = {arXiv preprint arXiv:2207.05561},
    year    = {2022}
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

