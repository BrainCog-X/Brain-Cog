# BrainCog

---

BrainCog is an open source spiking neural network based brain-inspired 
cognitive intelligence engine for Brain-inspired Artificial Intelligence, Brain-inspired Embodied AI, and brain simulation. More information on BrainCog can be found on its homepage http://www.brain-cog.network/

The current version of BrainCog contains at least 50 functional spiking neural network algorithms (including but not limited to perception and learning, decision making, knowledge representation and reasoning, motor control, social cognition, etc.) built based on BrainCog infrastructures, and BrainCog also provide brain simulations to drosophila, rodent, monkey, and human brains at multiple scales based on spiking neural networks at multiple scales. More detail in http://www.brain-cog.network/docs/

BrainCog is a community based effort for spiking neural network based artificial intelligence, and we welcome any forms of contributions, from contributing to the development of core components, to contributing for applications.

<img src="http://braincog.ai/static_index/image/github_readme/logo.jpg" alt="./figures/logo.jpg" width="70%" />

BrainCog provides essential and fundamental components to model biological and artificial intelligence.

![image]( http://braincog.ai/static_index/image/github_readme/braincog.png)

Our paper is published in [Patterns](https://www.cell.com/patterns/fulltext/S2666-3899(23)00144-7?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS2666389923001447%3Fshowall%3Dtrue). If you use BrainCog in your research, the following paper can be cited as the source for BrainCog.
```bib
@article{Zeng2023,
  doi = {10.1016/j.patter.2023.100789},
  url = {https://doi.org/10.1016/j.patter.2023.100789},
  year = {2023},
  month = jul,
  publisher = {Cell Press},
  pages = {100789},
  author = {Yi Zeng and Dongcheng Zhao and Feifei Zhao and Guobin Shen and Yiting Dong and Enmeng Lu and Qian Zhang and Yinqian Sun and Qian Liang and Yuxuan Zhao and Zhuoya Zhao and Hongjian Fang and Yuwei Wang and Yang Li and Xin Liu and Chengcheng Du and Qingqun Kong and Zizhe Ruan and Weida Bi},
  title = {{BrainCog}: A spiking neural network based,  brain-inspired cognitive intelligence engine for brain-inspired {AI} and brain simulation},
  journal = {Patterns}
}
```

## Brain-Inspired AI
BrainCog currently provides cognitive functions components that can be classified 
into five categories: 
* Perception and Learning
* Knowledge Representation and Reasoning
* Decision Making
* Motor Control
* Social Cognition
* Development and Evolution
* Safety and Security


<img src="https://raw.githubusercontent.com/Brain-Cog-Lab/Brain-Cog/main/figures/mirror-test.gif" alt="mt" width="55%" />
<img src="https://raw.githubusercontent.com/Brain-Cog-Lab/Brain-Cog/main/figures/joy.gif" alt="mt" width="55%" />

## Brain Simulation
BrainCog currently include two parts for brain simulation:
* Brain Cognitive Function Simulation
* Multi-scale Brain Structure Simulation


<img src="https://raw.githubusercontent.com/Brain-Cog-Lab/Brain-Cog/main/figures/braincog-mouse-brain-model-10s.gif" alt="bmbm10s" width="55%" /> 
<img src="https://raw.githubusercontent.com/Brain-Cog-Lab/Brain-Cog/main/figures/braincog-macaque-10s.gif" alt="bm10s" width="55%" />
<img src="https://raw.githubusercontent.com/Brain-Cog-Lab/Brain-Cog/main/figures/braincog-humanbrain-10s.gif" alt="bh10s" width="55%" />

The anatomical and imaging data is used to support our simulation from various aspects. 

## Software-Hardware Codesign (BrainCog Firefly)
BrainCog currently provides `hardware acceleration` for spiking neural network based brain-inspired AI.

<img src="http://braincog.ai/static_index/image/github_readme/firefly.jpg" alt="bh10s" width="55%" />

The following papers are most recent advancement of BrainCog Firefly series for Software-Hardware Codesign for Brain-inspired AI.
* Tenglong Li, Jindong Li, Guobin Shen, Dongcheng Zhao, Qian Zhang, Yi Zeng. FireFly-S: Exploiting Dual-Side Sparsity for Spiking Neural Networks Acceleration With Reconfigurable Spatial Architecture. IEEE Transactions on Circuits and Systems I (TCAS-I), 2024.(https://doi.org/10.1109/TCSI.2024.3496554)
* Jindong Li, Guobin Shen, Dongcheng Zhao, Qian Zhang, Yi Zeng. Firefly v2: Advancing hardware support for high-performance spiking neural network with a spatiotemporal fpga accelerator. IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 2024. (https://ieeexplore.ieee.org/abstract/document/10478105/)
* Jindong Li, Guobin Shen, Dongcheng Zhao, Qian Zhang, Yi Zeng. FireFly: A High-Throughput Hardware Accelerator for Spiking Neural Networks With Efficient DSP and Memory Optimization. IEEE Transactions on Very Large Scale Integration (VLSI) Systems, 2023. (https://ieeexplore.ieee.org/document/10143752)

## Embodied AI and Robotics (BrainCog Embot)
BrainCog Embot is an Embodied AI platform under the Brain-inspired Cognitive Intelligence Engine (BrainCog) framework, which is an open-source Brain-inspired AI platform based on Spiking Neural Network.
The following papers are most recent advancement of BrainCog Embot:
* Qianhao Wang, Yinqian Sun, Enmeng Lu, Qian Zhang, Yi Zeng. Brain-Inspired Action Generation with Spiking Transformer Diffusion Policy Model. Advances in Brain Inspired Cognitive Systems (BICS), 2024.(https://link.springer.com/chapter/10.1007/978-981-96-2882-7_23)
* Yinqian Sun, Feifei Zhao, Mingyang Lv, Yi Zeng. Implementing Spiking World Model with Multi-Compartment Neurons for Model-based Reinforcement Learning, 2025. (https://arxiv.org/abs/2503.00713)

## Resources
### [[Lectures]](https://github.com/BrainCog-X/Brain-Cog/blob/main/documents/Lectures.md)  |   [[Tutorial]](https://github.com/BrainCog-X/Brain-Cog/blob/main/documents/Tutorial.md)


## Publications using BrainCog 
### [[Brain Inspired AI]](https://github.com/BrainCog-X/Brain-Cog/blob/main/documents/Publication.md) | [[Brain Simulation]](https://github.com/BrainCog-X/Brain-Cog/blob/main/documents/Pub_brain_simulation.md) | [[Software-Hardware Co-design]](https://github.com/BrainCog-X/Brain-Cog/blob/main/documents/Pub_sh_codesign.md)

## BrainCog Data Engine
###  [BrainCog Data Engine](https://github.com/BrainCog-X/Brain-Cog/blob/main/documents/Data_engine.md)


## Requirements:
* numpy
* scipy
* h5py
* torch
* torchvision
* torchaudio
* timm == 0.6.13
* scikit-learn
* einops
* thop
* pyyaml
* matplotlib
* seaborn
* pygame
* dv
* tensorboard
* tonic



## Install 



### Install Online

1. You can install braincog by running:

    > `pip install braincog`

2. Also, install from github by running:

    > `pip install git+https://github.com/braincog-X/Brain-Cog.git`


### Install locally

1.  If you are a developer, it is recommanded to download or clone
    braincog from github.

    > `git clone https://github.com/braincog-X/Brain-Cog.git`

2.  Enter the folder of braincog

    > `cd Brain-Cog`

3.  Install braincog locally

    > `pip install -e .`
 

## Example 

1. Examples for Image Classification
```shell 
cd ./examples/Perception_and_Learning/img_cls/bp 
python main.py --model cifar_convnet --dataset cifar10 --node-type LIFNode --step 8 --device 0
```

2. Examples for Event Classification 

```shell
cd ./examples/Perception_and_Learning/img_cls/bp 
python main.py --model dvs_convnet --node-type LIFNode --dataset dvsc10 --step 10 --batch-size 128 --act-fun QGateGrad --device 0 
```
      
Other BrainCog features and tutorials can be found at http://www.brain-cog.network/docs/

## BrainCog Assistant 
Please add our BrainCog Assitant via wechat and we will invite you to our wechat developer group.
![image](https://github.com/Brain-Cog-Lab/Brain-Cog/blob/main/figures/wechat_ass.jpg)


## Maintenance
This project is led by 

**1.Brain-inspired Cognitive Intelligence Lab, Institute of Automation, Chinese Academy of Sciences http://www.braincog.ai/**

**2.Center for Long-term Artificial Intelligence (CLAI) http://long-term-ai.center/**
