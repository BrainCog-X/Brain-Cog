# BrainCog
BrainCog is an open source spiking neural network based brain-inspired 
cognitive intelligence engine for Brain-inspired Artificial Intelligence and brain simulation. More information on BrainCog can be found on its homepage http://www.brain-cog.network/

The current version of BrainCog contains at least 18 functional spiking neural network algorithms (including but not limited to perception and learning, decision making, knowledge representation and reasoning, motor control, social cognition, etc.) built based on BrainCog infrastructures, and BrainCog also provide brain simulations to drosophila, rodent, monkey, and human brains at multiple scales based on spiking neural networks at multiple scales. More detail in http://www.brain-cog.network/docs/

BrainCog is a community based effort for spiking neural network based artificial intelligence, and we welcome any forms of contributions, from contributing to the development of core components, to contributing for applications.

<img src="http://braincog.ai/static_index/image/github_readme/logo.jpg" alt="./figures/logo.jpg" width="70%" />

BrainCog provides essential and fundamental components to model biological and artificial intelligence.

![image]( http://braincog.ai/static_index/image/github_readme//braincog.png)

Our paper has been accepted by [Patterns](https://www.cell.com/patterns/fulltext/S2666-3899(23)00144-7?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS2666389923001447%3Fshowall%3Dtrue) recently. If you use BrainCog in your research, the following paper can be cited as the source for BrainCog.
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
* Decision Making
* Motor Control
* Knowledge Representation and Reasoning
* Social Cognition
 
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

## Resources
### [Lectures](https://github.com/BrainCog-X/Brain-Cog/tree/main/documents/Lectures.md)    [Tutorial](https://github.com/BrainCog-X/Brain-Cog/tree/main/documents/Tutorial.md)


## BrainCog Data Engine
In addition to the static datasets, BrainCog supports the commonly used neuromorphic
datasets, such as DVSGesture, DVSCIFAR10, NCALTECH101, ES-ImageNet.
Also, the neuromorphic dataset N-Omniglot for few-shot learning is also integrated into 
BrainCog.

**[DVSGesture](https://openaccess.thecvf.com/content_cvpr_2017/papers/Amir_A_Low_Power_CVPR_2017_paper.pdf)**

This dataset contains 11 hand gestures from 29 subjects under 3 illumination conditions recorded using a DVS128. 

**[DVSCIFAR10](https://www.frontiersin.org/articles/10.3389/fnins.2017.00309/full)**

This dataset converts 10,000 frame-based images in the CIFAR10 dataset into 10,000 event streams using a dynamic vision sensor.

**[NCALTECH101](https://www.frontiersin.org/articles/10.3389/fnins.2015.00437/full)**

The NCaltech101 dataset is captured by mounting the ATIS sensor on a motorized pan-tilt unit and having the sensor move while it views Caltech101 examples on an LCD monitor. 
The "Faces" class has been removed from N-Caltech101, leaving 100 object classes plus a background class

**[ES-ImageNet](https://arxiv.org/abs/2110.12211)**

The dataset is converted with Omnidirectional Discrete Gradient (ODG) from 1,300,000 frame-based images in the ImageNet dataset into event-stream samples, which has 1000 categories. 

**[N-Omniglot](https://www.nature.com/articles/s41597-022-01851-z)**

This dataset contains 1,623 categories of handwritten characters, with only 20 samples per class. 
The dataset is acquired with the DVS acquisition platform to shoot videos (generated from the original Omniglot dataset) played on the monitor, and use the Robotic Process Automation (RPA) software to collect the data automatically.
 
You can easily use them in the braincog/datasets folder, taking DVSCIFAR10 as an example
```python
loader_train, loader_eval,_,_ = get_dvsc10_data(batch_size=128,step=10)
```

## Requirements:
* python == 3.8
* CUDA toolkit == 11.
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
* tonic
* pandas >= 1.4.2  
* xlrd == 1.2.0


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

# Publications Using BrainCog 
## Brain Inspired AI
### Perception and Leanring
| Papers                                                                                                                                                                                 | Codes                                                                                                              | Publisher                              |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|----------------------------------------|
| [Quantum superposition inspired spiking neural network](https://www.cell.com/iscience/fulltext/S2589-0042(21)00848-8)                                                                  | https://github.com/BrainCog-X/Brain-Cog/tree/main/examples/Perception_and_Learning/QSNN                            | Cell iScience                          |
| [Backpropagation with biologically plausible spatiotemporal adjustment for training deep spiking neural networks](https://www.cell.com/patterns/pdf/S2666-3899(22)00119-2.pdf)         | https://github.com/BrainCog-X/Brain-Cog/tree/main/examples/Perception_and_Learning/img_cls/bp                      | Cell Patterns                          |
| [N-Omniglot, a large-scale neuromorphic dataset for spatio-temporal sparse few-shot learning](https://www.nature.com/articles/s41597-022-01851-z)                                      | https://github.com/BrainCog-X/Brain-Cog/tree/main/braincog/datasets/NOmniglot                                      | Scientific Data                        |
| [Efficient and Accurate Conversion of Spiking Neural Network with Burst Spikes](https://www.ijcai.org/proceedings/2022/0345.pdf)                                                       | https://github.com/BrainCog-X/Brain-Cog/blob/main/examples/Perception_and_Learning/Conversion/converted_CIFAR10.py | IJCAI 2022                             |
| [BackEISNN: A deep spiking neural network with adaptive self-feedback and balanced excitatory–inhibitory neurons](https://www.sciencedirect.com/science/article/pii/S0893608022002520) | https://github.com/BrainCog-X/Brain-Cog/blob/main/examples/Perception_and_Learning/img_cls/bp/main_backei.py       | Neural Networks                        |
| [Spiking CapsNet: A spiking neural network with a biologically plausible routing rule between capsules](https://www.sciencedirect.com/science/article/pii/S002002552200843X)           | https://github.com/BrainCog-X/Brain-Cog/tree/main/examples/Perception_and_Learning/img_cls/spiking_capsnet         | Information Sciences                   |
| [Multisensory Concept Learning Framework Based on Spiking Neural Networks](https://www.frontiersin.org/articles/10.3389/fnsys.2022.845177/full)                                        | https://github.com/BrainCog-X/Brain-Cog/tree/main/examples/Perception_and_Learning/MultisensoryIntegration         | Frontiers in Systems Neuroscience      |
| [GLSNN: A Multi-Layer Spiking Neural Network Based on Global Feedback Alignment and Local STDP Plasticity](https://www.frontiersin.org/articles/10.3389/fncom.2020.576841/full)        | https://github.com/BrainCog-X/Brain-Cog/tree/main/examples/Perception_and_Learning/img_cls/glsnn                   | Fontiers in Computational Neuroscience |
| [EventMix: An Efficient Augmentation Strategy for Event-Based Data](https://arxiv.org/abs/2205.12054)                                                                                  | https://github.com/BrainCog-X/Brain-Cog/blob/main/braincog/datasets/cut_mix.py                                     | Arxiv                                  |
| [Spike Calibration: Fast and Accurate Conversion of Spiking Neural Network for Object Detection and Segmentation](https://arxiv.org/abs/2207.02702)                                    | https://github.com/BrainCog-X/Brain-Cog/blob/main/examples/Perception_and_Learning/Conversion/converted_CIFAR10.py | Arxiv                                  |
| [An Unsupervised Spiking Neural Network Inspired By Biologically Plausible Learning Rules and Connections](https://arxiv.org/abs/2207.02727)                                           | https://github.com/BrainCog-X/Brain-Cog/tree/main/examples/Perception_and_Learning/UnsupervisedSTDP                | Arxiv                                  |

### Social Cognition
| Papers                                                                                                                                                                    | Codes                                                                                                  | Publisher                               |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|-----------------------------------------|
| [Toward Robot Self-Consciousness (II): Brain-Inspired Robot Bodily Self Model for Self-Recognition](https://link.springer.com/article/10.1007/s12559-017-9505-1 )         | https://github.com/BrainCog-X/Brain-Cog/tree/main/examples/Social_Cognition/mirror_test                | Cognitive Computation                   |
| [A brain-inspired intention prediction model and its applications to humanoid robot](https://www.frontiersin.org/articles/10.3389/fnins.2022.1009237/full)                | https://github.com/BrainCog-X/Brain-Cog/tree/main/examples/Social_Cognition/Intention_Prediction       | Frontiers in Neuroscience               |
| [A Brain-Inspired Theory of Mind Spiking Neural Network for Reducing Safety Risks of Other Agents](https://www.frontiersin.org/articles/10.3389/fnins.2022.753900/full)   | https://github.com/BrainCog-X/Brain-Cog/tree/main/examples/Social_Cognition/ToM	                       | Frontiers in Neuroscience               |
| [Brain-Inspired Affective Empathy Computational Model and Its Application on Altruistic Rescue Task](https://www.frontiersin.org/articles/10.3389/fncom.2022.784967/full) | https://github.com/BrainCog-X/Brain-Cog/tree/main/examples/Social_Cognition/affective_empathy/BAE-SNN  | Frontiers in Computational Neuroscience |
| [A brain-inspired robot pain model based on a spiking neural network](https://www.frontiersin.org/articles/10.3389/fnbot.2022.1025338/full)                               | https://github.com/BrainCog-X/Brain-Cog/tree/main/examples/Social_Cognition/affective_empathy/BRP-SNN	 | Frontiers in Neurorobotics              |
| [Brain-Inspired Theory of Mind Spiking Neural Network Elevates Multi-Agent Cooperation and Competition](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4271099)      | https://github.com/BrainCog-X/Brain-Cog/tree/main/examples/Social_Cognition/MAToM-SNN                  | SSRN                                    |

### Knowledge Representation and Reasoning
| Papers                                                                                                                                                                                | Codes                                                                                                          | Publisher                               |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|-----------------------------------------|
| [A Brain-Inspired Causal Reasoning Model Based on Spiking Neural Networks](https://ieeexplore.ieee.org/abstract/document/9534102)                                                     | https://github.com/BrainCog-X/Brain-Cog/tree/main/examples/Knowledge_Representation_and_Reasoning/CRSNN        | IJCNN2021                               |
| [Brain Inspired Sequences Production by Spiking Neural Networks With Reward-Modulated STDP](https://www.frontiersin.org/articles/10.3389/fncom.2021.612041/full)                      | https://github.com/BrainCog-X/Brain-Cog/tree/main/examples/Knowledge_Representation_and_Reasoning/SPSNN        | Frontiers in Computational Neuroscience |
| [Temporal-Sequential Learning With a Brain-Inspired Spiking Neural Network and Its Application to Musical Memory](https://www.frontiersin.org/articles/10.3389/fncom.2020.00051/full) | https://github.com/BrainCog-X/Brain-Cog/tree/main/examples/Knowledge_Representation_and_Reasoning/musicMemory  | Frontiers in Computational Neuroscience |
| [Brain-Inspired Affective Empathy Computational Model and Its Application on Altruistic Rescue Task](https://www.frontiersin.org/articles/10.3389/fncom.2022.784967/full)             | https://github.com/BrainCog-X/Brain-Cog/tree/main/examples/Knowledge_Representation_and_Reasoning/musicMemory  | Frontiers in System Neuroscience        |
| [Stylistic Composition of Melodies Based on a Brain-Inspired Spiking Neural Network](https://www.frontiersin.org/articles/10.3389/fnsys.2021.639484/full)                             | https://github.com/BrainCog-X/Brain-Cog/tree/main/examples/Social_Cognition/affective_empathy/BRP-SNN	         | Frontiers in Neurorobotics              |
| [Brain-inspired Graph Spiking Neural Networks for Commonsense Knowledge Representation and Reasoning](https://arxiv.org/abs/2207.05561 )                                              | https://github.com/BrainCog-X/Brain-Cog/tree/main/examples/Knowledge_Representation_and_Reasoning/CKRGSNN      | Arxiv                                   |

### Decision Making
| Papers                                                                                                                                                                                      | Codes                                                                                                          | Publisher                        |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|----------------------------------|
| [Nature-inspired self-organizing collision avoidance for drone swarm based on reward-modulated spiking neural network](https://www.cell.com/patterns/fulltext/S2666-3899(22)00236-7)        | https://github.com/BrainCog-X/Brain-Cog/blob/main/examples/decision_making/swarm/Collision-Avoidance.py        | Cell Patterns                    |
| [Solving the spike feature information vanishing problem in spiking deep Q network with potential based normalization](https://www.frontiersin.org/articles/10.3389/fnins.2022.953368/full) | https://github.com/BrainCog-X/Brain-Cog/tree/main/examples/decision_making/RL/sdqn                             | Frontiers in Neuroscience        |
| [A Brain-Inspired Decision-Making Spiking Neural Network and Its Application in Unmanned Aerial Vehicle](https://www.frontiersin.org/articles/10.3389/fnbot.2018.00056/full )               | https://github.com/BrainCog-X/Brain-Cog/blob/main/examples/decision_making/BDM-SNN/BDM-SNN-hh.py	              | Frontiers in Neurorobotics       |
| [Multi-compartment Neuron and Population Encoding improved Spiking Neural Network for Deep Distributional Reinforcement Learning](https://arxiv.org/abs/2301.07275)                         | https://github.com/BrainCog-X/Brain-Cog/tree/main/examples/decision_making/RL/mcs-fqf                          | Arxiv                            |

### Motor Control
| Papers | Codes                                                                                          | Publisher |
|--------|------------------------------------------------------------------------------------------------|-----------|
|        | https://github.com/BrainCog-X/Brain-Cog/tree/main/examples/MotorControl/experimental	          |           |


### SNN Safety
| Papers | Codes                                                                        | Publisher |
|--------|------------------------------------------------------------------------------|-----------|
| DPSNN  | https://github.com/BrainCog-X/Brain-Cog/tree/main/examples/Snn_safety/DPSNN	 | Arxiv     |

### Development and Evolution
| Papers                                                                                                                                 | Codes                                                                                        | Publisher |
|----------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|-----------|
| [Developmental Plasticity-inspired Adaptive Pruning for Deep Spiking and Artificial Neural Networks](https://arxiv.org/abs/2211.12714) | https://github.com/BrainCog-X/Brain-Cog/tree/main/examples/Structural_Development/DPAP	      | Arxiv     |
| [Adaptive Sparse Structure Development with Pruning and Regeneration for Spiking Neural Networks](https://arxiv.org/abs/2211.12219)    | https://github.com/BrainCog-X/Brain-Cog/tree/main/examples/Structural_Development/SD-SNN     | Arxiv     |

### Hardware Acceleration
| Papers                                                                                                                                              | Codes                                                                                                         | Publisher |
|-----------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|-----------|
| [FireFly: A High-Throughput and Reconfigurable Hardware Accelerator for Spiking Neural Networks](https://arxiv.org/abs/2301.01905)                  | https://github.com/BrainCog-X/Brain-Cog/tree/main/examples/Hardware_acceleration	                             | Arxiv     |


## Brain Simulation
### Funtion

| Papers                                                                                                                                                                                                       | Codes                                                                                                                     | Publisher |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|-----------|
| [A neural algorithm for Drosophila linear and nonlinear decision-making](https://www.nature.com/articles/s41598-020-75628-y)                                                                                 | https://github.com/BrainCog-X/Brain-Cog/blob/main/examples/Brain_Cognitive_Function_Simulation/drosophila/drosophila.py	  | Scientific Reports|
| [Comparison Between Human and Rodent Neurons for Persistent Activity Performance: A Biologically Plausible Computational Investigation](https://www.frontiersin.org/articles/10.3389/fnsys.2021.628839/full) | https://github.com/BrainCog-X/Brain-Cog/tree/main/examples/Multiscale_Brain_Structure_Simulation/Human_PFC_Model	         | Frontiers in System Neuroscience|

### Structure

| Papers | Codes                                                                                                                         | Publisher |
|--------|-------------------------------------------------------------------------------------------------------------------------------|-----------|
|   Corticothalamic minicolumn	     | 		https://github.com/BrainCog-X/Brain-Cog/tree/main/examples/Multiscale_Brain_Structure_Simulation/CorticothalamicColumn      | |
|   Human Brain	     | https://github.com/BrainCog-X/Brain-Cog/tree/main/examples/Multiscale_Brain_Structure_Simulation/HumanBrain		                 | |
|    Macaque Brain    | https://github.com/BrainCog-X/Brain-Cog/tree/main/examples/Multiscale_Brain_Structure_Simulation/MacaqueBrain	                | |
|    Mouse Brain    | https://github.com/BrainCog-X/Brain-Cog/tree/main/examples/Multiscale_Brain_Structure_Simulation/Mouse_brain 	                | |
