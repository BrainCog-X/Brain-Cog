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

## Input:

The 88 regions' connectivity matrix can be obtained from the following link:
[https://drive.google.com/file/d/1f8fpXgR8X07HrJ7G9DwMAl8K0naPcxJC/view?usp=sharing](https://drive.google.com/file/d/1tLHxCtm2kawKVvJ1BhAbkFKeyxcrJwnO/view?usp=sharing)

The source of this connectivity matrix is in the following link:
https://www.nitrc.org/frs/?group_id=432

## Example:

```shell 
cd ~/examples/Multi-scale Brain Structure Simulation/HumanBrain/
python human_brain.py
```

## Parameters:
The parameters are similar to mouse brain simulation 


