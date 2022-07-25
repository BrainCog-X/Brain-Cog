Music Memory and stylistic composition

This repository contains code from our paper:
- [**Temporal-Sequential Learning With a Brain-Inspired Spiking Neural Network and Its Application to Musical Memory**](https://www.cell.com/patterns/fulltext/S2666-3899(22)00119-2) published in Frontiers in Computational Neuroscience. **https://www.cell.com/patterns/fulltext/S2666-3899(22)00119-2**,
- [**Stylistic composition of melodies based on a brain-inspired spiking neural network**](https://www.frontiersin.org/articles/10.3389/fnsys.2021.639484/full) published in  Frontiers in Systems Neuroscience **https://www.frontiersin.org/articles/10.3389/fnsys.2021.639484/full**.


## Requirments

* numpy
* scipy
* pytorch >= 1.7.0
* pretty_midi >= 0.2.9


## Data preparation

The dataset used here can be referred to the website http://www.piano-midi.de/. 


## Run
* Run the script *task/musicMemory.py* to memorize and recall the musical melodies, the result will be recorded in a midi file.
* Run the script *task/musicGeneration.py* to learn and generate melodies with different styles, the result will be recorded in a midi file.
The API and details can be found in these scripts. 

## Citation
If you use our code or refer to this project, please cite these papers.

```BibTex
@article{zeng2018toward,
  title={Toward robot self-consciousness (ii): brain-inspired robot bodily self model for self-recognition},
  author={Zeng, Yi and Zhao, Yuxuan and Bai, Jun and Xu, Bo},
  journal={Cognitive Computation},
  volume={10},
  number={2},
  pages={307--320},
  year={2018},
  publisher={Springer}
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
