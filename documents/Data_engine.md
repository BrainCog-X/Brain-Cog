# BrainCog Data Engine

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