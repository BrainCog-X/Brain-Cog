# Human Brain Simulation

## Description
Human Brain Simulation is a large scale brain modeling framework depending on braincog framework.

## Requirements:
* numpy >= 1.21.2
* scipy >= 1.8.0
* h5py >= 3.6.0
* torch >= 1.10
* torchvision >= 0.12.0
* torchaudio  >= 0.11.0
* timm >= 0.5.4
* matplotlib >= 3.5.1
* einops >= 0.4.1
* thop >= 0.0.31
* pyyaml >= 6.0
* loris >= 0.5.3
* pandas >= 1.4.2  
* tonic (special)
* pandas >= 1.4.2  

## Example:

```shell 
cd ~/examples/Multi-scale Brain Structure Simulation/HumanBrain/
python brainSimHum.py
```

## Parameters:
To simulate the models (both human and macaque brain), the parameters of the neuron number in each region and the connectome power between regions can be set flexibly in the main function (nsz and asz) of the .py files.

## Citations:
If you find this package helpful, please consider citing the following papers:

    @article{Liu2016,
    author={Liu, Xin and Zeng, Yi and Zhang, Tielin and Xu, Bo},
    title={Parallel Brain Simulator: A Multi-scale and Parallel Brain-Inspired Neural Network Modeling and Simulation Platform},
    journal={Cognitive Computation},
    year={2016},
    month={Oct},
    day={01},
    volume={8},
    number={5},
    pages={967--981},
    issn={1866-9964},
    doi={10.1007/s12559-016-9411-y},
    url={https://doi.org/10.1007/s12559-016-9411-y}
    }

    @misc{https://doi.org/10.48550/arxiv.2207.08533,
      doi = {10.48550/ARXIV.2207.08533},
      url = {https://arxiv.org/abs/2207.08533},
      author = {Zeng, Yi and Zhao, Dongcheng and Zhao, Feifei and Shen, Guobin and Dong, Yiting and Lu, Enmeng and Zhang, Qian and Sun, Yinqian and Liang, Qian and Zhao, Yuxuan and Zhao, Zhuoya and Fang, Hongjian and Wang, Yuwei and Li, Yang and Liu, Xin and Du, Chengcheng and Kong, Qingqun and Ruan, Zizhe and Bi, Weida},
      title = {BrainCog: A Spiking Neural Network based Brain-inspired Cognitive Intelligence Engine for Brain-inspired AI and Brain Simulation},
      publisher = {arXiv},
      year = {2022},
    }
