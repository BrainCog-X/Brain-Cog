# PL-SDQN

This repository contains code from our paper [**Solving the Spike Feature Information Vanishing Problem in Spiking Deep Q Network with Potential Based Normalization**]. If you use our code or refer to this project, please cite this paper.
To run the PL-SDQN model, please install 'tianshou' framework first https://github.com/thu-ml/tianshou

## Requirments

* numpy
* scipy
* pytorch >= 1.7.0
* torchvision
* gymnasium[atari, accept-rom-license]
* atari-py
* opencv-python
* tianshou

## Train

```shell  
python ./main_sdqn.py
```

or

```shell
python ./main_mcs_fqf.py
```


## Citation

If you find this package helpful, please consider citing the following papers:

```BibTex
@ARTICLE{sun2022,
AUTHOR={Sun, Yinqian and Zeng, Yi and Li, Yang},    
TITLE={Solving the spike feature information vanishing problem in spiking deep Q network with potential based normalization},      
JOURNAL={Frontiers in Neuroscience},      
VOLUME={16},           
YEAR={2022},      	  
URL={https://www.frontiersin.org/articles/10.3389/fnins.2022.953368},       
DOI={10.3389/fnins.2022.953368},      
ISSN={1662-453X},   
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
