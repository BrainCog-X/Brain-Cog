# Drosophila-inspired decision-making SNN

## Run

The drosophila.py implements the core code of the Drosophila-inspired linear and non-linear decision-making in paper entitled "A Neural Algorithm for linear and non-linear Decision-making inspired by Drosophila".

The experiments includes training phase and testing phase:

* Training Phase

Training linear network and nonlinear network by reward-modulated spiking neural network: green-upright T is safe and blue-inverted T is dangerous

* Testing Phase 

For linear pathway and nonlinear pathway, choose between blue-upright T and green-inverted T, and count the PI values under different color intensity

## Results

The following picture shows the linear (a) and nonlinear (b) pathways, the training and testing phases (c), and the PI values on different color intensities (d).

![description](./dro.jpg)

Differences from the original article: an improved reward-modulated STDP learning rule. 

## Citation

If you find this package helpful, please consider citing the following papers:

```BibTex
@article{zhao2020neural,
  title={A neural algorithm for Drosophila linear and nonlinear decision-making},
  author={Zhao, Feifei and Zeng, Yi and Guo, Aike and Su, Haifeng and Xu, Bo},
  journal={Scientific Reports},
  volume={10},
  number={1},
  pages={1--16},
  year={2020},
  publisher={Nature Publishing Group}
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
