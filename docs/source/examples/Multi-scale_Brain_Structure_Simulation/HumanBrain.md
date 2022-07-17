# Human Brain Simulation

## Description
Human Brain Simulation is a large scale brain modeling framework depending on BrainCog framework.

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
