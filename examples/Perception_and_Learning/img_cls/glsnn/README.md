# SNN with global feedback connections
Training deep spiking neural network with the global 
feedback connections and the local optimization learning rules. And is a little different from our original paper.

GLSNN: A Multi-layer Spiking Neural Network based on Global Feedback Alignment and Local STDP Plasticity.

## Results
```shell
python cls_glsnn.py
```
We train the model for 100 epochs, and the best accuracy for MNIST is 98.23\%, for FashionMNIST is 89.68\%.
![image](result_zdc.png)

## Citation

If you find the code and dataset useful in your research, please consider citing:
```
@article{zhao2020glsnn,
  title={GLSNN: A Multi-Layer Spiking Neural Network Based on Global Feedback Alignment and Local STDP Plasticity},
  author={Zhao, Dongcheng and Zeng, Yi and Zhang, Tielin and Shi, Mengting and Zhao, Feifei},
  journal={Frontiers in Computational Neuroscience},
  volume={14},
  year={2020},
  publisher={Frontiers Media SA}
}
```
## Contents
Feedbacks and comments are welcome! Feel free to contact us via [zhaodongcheng2016@ia.ac.cn](zhaodongcheng2016@ia.ac.cn) 

Enjoy!