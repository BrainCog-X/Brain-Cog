# Conversion Method
Training deep spiking neural network with ann-snn conversion
replace ReLU and MaxPooling in pytorch model to make origin ANN to be converted SNN to finish complex tasks

## Results
```shell
python CIFAR10_VGG16.py
python converted_CIFAR10.py
```

You should first run the `CIFAR10_VGG16.py` to get a well-trained ANN.
Then `converted_CIFAR10.py` can be used to run the snn inference process.

### Citation 
If you find this package helpful, please consider citing it:

```BibTex
@inproceedings{ijcai2022p345,
  title     = {Efficient and Accurate Conversion of Spiking Neural Network with Burst Spikes},
  author    = {Li, Yang and Zeng, Yi},
  booktitle = {Proceedings of the Thirty-First International Joint Conference on
               Artificial Intelligence, {IJCAI-22}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  pages     = {2485--2491},
  year      = {2022},
  month     = {7},
}


@article{li2022spike,
title={Spike calibration: Fast and accurate conversion of spiking neural network for object detection and segmentation},
author={Li, Yang and He, Xiang and Dong, Yiting and Kong, Qingqun and Zeng, Yi},
journal={arXiv preprint arXiv:2207.02702},
year={2022}
}

```