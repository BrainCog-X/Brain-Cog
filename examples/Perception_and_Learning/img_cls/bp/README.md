# Script for training high-performance SNNs based on back propagation 
This is an example of training high-performance SNNs using the braincog.
It is able to train high performance SNNs on CIFAR10, DVS-CIFAR10, ImageNet and other datasets, and reach the advanced level. 

## Install braincog  

```shell
git clone https://github.com/xxx/Brain-Cog.git
cd braincog 
python setup install --user 
```

## Examples of training

```shell
cd examples/img_cls/bp 
python main.py --model dvs_convnet --node-type LIFNode --dataset dvsc10 --step 10 --batch-size 128 --act-fun QGate --device 0 
```

## Benchmark 

We provide a benchmark of SNNs trained with braincog and the corresponding scripts. 
This provides an open, fair platform for comparison of subsequent SNNs on classification tasks. 

**Note**: The results may vary due to random seeding and software version issues. 

### CIFAR10 

| ID  | Dataset | Node-type  | Config |    Model    | Batch Size |   Accuracy   | Script                                                                                                                                     |
|:----|:-------:|:----------:|:------:|:-----------:|:----------:|:------------:|:-------------------------------------------------------------------------------------------------------------------------------------------|
| 1   | CIFAR10 |  IF+Atan   |   -    |   convnet   |    128     |    95.54     | ```python main.py --model cifar_convnet --node-type IFNode --dataset cifar10 --step 4 --batch-size 128 --act-fun AtanGrad --device 0```    |
| 1   | CIFAR10 |  LIF+Atan  |   -    |   convnet   |    128     |    91.92     | ```python main.py --model cifar_convnet --node-type LIFNode --dataset cifar10 --step 4 --batch-size 128 --act-fun AtanGrad --device 0```   |
| 1   | CIFAR10 | PLIF+Atan  |   -    |   convert   |    128     |    93.32     | ```python main.py --model cifar_convnet --node-type PLIFNode --dataset cifar10 --step 4 --batch-size 128 --act-fun AtanGrad --device 0```  |
| 1   | CIFAR10 |  IF+Atan   |   -    |  resnet18   |    128     | 89.76/89.80  | ```python main.py --model resnet18 --node-type IFNode --dataset cifar10 --step 4 --batch-size 128 --act-fun AtanGrad --device 0```         |
| 1   | CIFAR10 |  LIF+Atan  |   -    |  resnet18   |    128     | 89.93/89.88  | ```python main.py --model resnet18 --node-type LIFNode --dataset cifar10 --step 4 --batch-size 128 --act-fun AtanGrad --device 0```        |
| 1   | CIFAR10 | PLIF+Atan  |   -    |  resnet18   |    128     | 92.64/ 90.65 | ```python main.py --model resnet18 --node-type PLIFNode --dataset cifar10 --step 4 --batch-size 128 --act-fun AtanGrad --device 0```       |
| 1   | CIFAR10 |  IF+QGate  |   -    | dvs_convnet |    128     |    95.73     | ```python main.py --model cifar_convnet --node-type IFNode --dataset cifar10 --step 4 --batch-size 128 --act-fun QGateGrad --device 0```   |
| 1   | CIFAR10 | LIF+QGate  |   -    | dvs_convnet |    128     |    96.04     | ```python main.py --model cifar_convnet --node-type LIFNode --dataset cifar10 --step 4 --batch-size 128 --act-fun QGateGrad --device 0```  |
| 1   | CIFAR10 | PLIF+QGate |   -    | dvs_convnet |    128     | 96.04/95.84  | ```python main.py --model cifar_convnet --node-type PLIFNode --dataset cifar10 --step 4 --batch-size 128 --act-fun QGateGrad --device 0``` |
| 1   | CIFAR10 |  IF+QGate  |   -    |  resnet18   |    128     |    89.19     | ```python main.py --model resnet18 --node-type IFNode --dataset cifar10 --step 4 --batch-size 128 --act-fun QGateGrad --device 0```        |
| 1   | CIFAR10 | LIF+QGate  |   -    |  resnet18   |    128     | 90.95/90.68  | ```python main.py --model resnet18 --node-type LIFNode --dataset cifar10 --step 4 --batch-size 128 --act-fun QGateGrad --device 0```       |
| 1   | CIFAR10 | PLIF+QGate |   -    |  resnet18   |    128     | 90.97/91.02  | ```python main.py --model resnet18 --node-type PLIFNode --dataset cifar10 --step 4 --batch-size 128 --act-fun QGateGrad --device 0```      |


### CIFAR100 
| ID  | Dataset  | Node-type  | Config |    Model    | Batch Size | Accuracy | Script                                                                                                                                                        |
|:----|:--------:|:----------:|:------:|:-----------:|:----------:|:--------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1   | CIFAR100 |  IF+Atan   |   -    | dvs_convnet |    128     |  76.52   | ```python main.py --num-classes 100 --model cifar_convnet --node-type IFNode --dataset cifar100 --step 4 --batch-size 128 --act-fun AtanGrad --device 0```    |
| 1   | CIFAR100 |  LIF+Atan  |   -    | dvs_convnet |    128     |  71.89   | ```python main.py --num-classes 100 --model cifar_convnet --node-type LIFNode --dataset cifar100 --step 4 --batch-size 128 --act-fun AtanGrad --device 0```   |
| 1   | CIFAR100 | PLIF+Atan  |   -    | dvs_convnet |    128     |  72.82   | ```python main.py --num-classes 100 --model cifar_convnet --node-type PLIFNode --dataset cifar100 --step 4 --batch-size 128 --act-fun AtanGrad --device 0```  |
| 1   | CIFAR100 |  IF+Atan   |   -    |  resnet18   |    128     |  62.47   | ```python main.py --num-classes 100 --model resnet18 --node-type IFNode --dataset cifar100 --step 4 --batch-size 128 --act-fun AtanGrad --device 0```         |
| 1   | CIFAR100 |  LIF+Atan  |   -    |  resnet18   |    128     |  62.63   | ```python main.py --num-classes 100 --model resnet18 --node-type LIFNode --dataset cifar100 --step 4 --batch-size 128 --act-fun AtanGrad --device 0```        |
| 1   | CIFAR100 | PLIF+Atan  |   -    |  resnet18   |    128     |  62.71   | ```python main.py --num-classes 100 --model resnet18 --node-type PLIFNode --dataset cifar100 --step 4 --batch-size 128 --act-fun AtanGrad --device 0```       |
| 1   | CIFAR100 |  IF+QGate  |   -    | dvs_convnet |    128     |  76.44   | ```python main.py --num-classes 100 --model cifar_convnet --node-type IFNode --dataset cifar100 --step 4 --batch-size 128 --act-fun QGateGrad --device 0```   |
| 1   | CIFAR100 | LIF+QGate  |   -    | dvs_convnet |    128     |  77.73   | ```python main.py --num-classes 100 --model cifar_convnet --node-type LIFNode --dataset cifar100 --step 4 --batch-size 128 --act-fun QGateGrad --device 0```  |
| 1   | CIFAR100 | PLIF+QGate |   -    | dvs_convnet |    128     |  77.25   | ```python main.py --num-classes 100 --model cifar_convnet --node-type PLIFNode --dataset cifar100 --step 4 --batch-size 128 --act-fun QGateGrad --device 0``` |
| 1   | CIFAR100 |  IF+QGate  |   -    |  resnet18   |    128     |  60.01   | ```python main.py --num-classes 100 --model resnet18 --node-type IFNode --dataset cifar100 --step 4 --batch-size 128 --act-fun QGateGrad --device 0```        |
| 1   | CIFAR100 | LIF+QGate  |   -    |  resnet18   |    128     |  61.33   | ```python main.py --num-classes 100 --model resnet18 --node-type LIFNode --dataset cifar100 --step 4 --batch-size 128 --act-fun QGateGrad --device 0```       |
| 1   | CIFAR100 | PLIF+QGate |   -    |  resnet18   |    128     |  62.32   | ```python main.py --num-classes 100 --model resnet18 --node-type PLIFNode --dataset cifar100 --step 4 --batch-size 128 --act-fun QGateGrad --device 0```      |


### DVS-CIFAR10

| ID  |   Dataset   | Node-type  | Config |    Model    | Batch Size | FLOPS |  Accuracy   | Script                                                                                                                                   |
|:----|:-----------:|:----------:|:------:|:-----------:|:----------:|:-----:|:-----------:|:-----------------------------------------------------------------------------------------------------------------------------------------|
| 1   | DVS-CIFAR10 |  IF+Atan   |   -    | dvs_convnet |    128     | 7503  |    65.90    | ```python main.py --model dvs_convnet --node-type IFNode --dataset dvsc10 --step 10 --batch-size 128 --act-fun AtanGrad --device 0```    |
| 1   | DVS-CIFAR10 |  LIF+Atan  |   -    | dvs_convnet |    128     | 7503  |    82.10    | ```python main.py --model dvs_convnet --node-type LIFNode --dataset dvsc10 --step 10 --batch-size 128 --act-fun AtanGrad --device 0```   |
| 1   | DVS-CIFAR10 | PLIF+Atan  |   -    | dvs_convnet |    128     | 7503  |    81.90    | ```python main.py --model dvs_convnet --node-type PLIFNode --dataset dvsc10 --step 10 --batch-size 128 --act-fun AtanGrad --device 0```  |
| 1   | DVS-CIFAR10 |  IF+Atan   |   -    |  resnet18   |    128     | 3149  |    69.10    | ```python main.py --model resnet18 --node-type IFNode --dataset dvsc10 --step 10 --batch-size 128 --act-fun AtanGrad --device 0```       |
| 1   | DVS-CIFAR10 |  LIF+Atan  |   -    |  resnet18   |    128     | 3149  |    78.50    | ```python main.py --model resnet18 --node-type LIFNode --dataset dvsc10 --step 10 --batch-size 128 --act-fun AtanGrad --device 0```      |
| 1   | DVS-CIFAR10 | PLIF+Atan  |   -    |  resnet18   |    128     | 3149  |    77.70    | ```python main.py --model resnet18 --node-type PLIFNode --dataset dvsc10 --step 10 --batch-size 128 --act-fun AtanGrad --device 0```     |
| 1   | DVS-CIFAR10 |  IF+QGate  |   -    | dvs_convnet |    128     | 7503  |    68.30    | ```python main.py --model dvs_convnet --node-type IFNode --dataset dvsc10 --step 10 --batch-size 128 --act-fun QGateGrad --device 0```   |
| 1   | DVS-CIFAR10 | LIF+QGate  |   -    | dvs_convnet |    128     | 7503  | 82.60/82.90 | ```python main.py --model dvs_convnet --node-type LIFNode --dataset dvsc10 --step 10 --batch-size 128 --act-fun QGateGrad --device 0```  |
| 1   | DVS-CIFAR10 | PLIF+QGate |   -    | dvs_convnet |    128     | 7503  |    83.20    | ```python main.py --model dvs_convnet --node-type PLIFNode --dataset dvsc10 --step 10 --batch-size 128 --act-fun QGateGrad --device 0``` |
| 1   | DVS-CIFAR10 |  IF+QGate  |   -    |  resnet18   |    128     | 3149  | 65.70/66.80 | ```python main.py --model resnet18 --node-type IFNode --dataset dvsc10 --step 10 --batch-size 128 --act-fun QGateGrad --device 0```      |
| 1   | DVS-CIFAR10 | LIF+QGate  |   -    |  resnet18   |    128     | 3149  | 79.00/79.40 | ```python main.py --model resnet18 --node-type LIFNode --dataset dvsc10 --step 10 --batch-size 128 --act-fun QGateGrad --device 0```     |
| 1   | DVS-CIFAR10 | PLIF+QGate |   -    |  resnet18   |    128     | 3149  | 78.10/78.20 | ```python main.py --model resnet18 --node-type PLIFNode --dataset dvsc10 --step 10 --batch-size 128 --act-fun QGateGrad --device 0```    |


### DVS-Gesture

| ID  | Dataset | Node-type  | Config |    Model    | Batch Size |  Accuracy   | Script                                                                                                                                                  |
|:----|:-------:|:----------:|:------:|:-----------:|:----------:|:-----------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1   |  DVS-G  |  IF+Atan   |   -    | dvs_convnet |    128     |    64.77    | ```python main.py --num-classes 11 --model dvs_convnet --node-type IFNode --dataset dvsg --step 10 --batch-size 128 --act-fun AtanGrad --device 0```    |
| 1   |  DVS-G  |  LIF+Atan  |   -    | dvs_convnet |    128     |    91.28    | ```python main.py --num-classes 11 --model dvs_convnet --node-type LIFNode --dataset dvsg --step 10 --batch-size 128 --act-fun AtanGrad --device 0```   |
| 1   |  DVS-G  | PLIF+Atan  |   -    | dvs_convnet |    128     |    91.67    | ```python main.py --num-classes 11 --model dvs_convnet --node-type PLIFNode --dataset dvsg --step 10 --batch-size 128 --act-fun AtanGrad --device 0```  |
| 1   |  DVS-G  |  IF+Atan   |   -    |  resnet18   |    128     |    63.25    | ```python main.py --num-classes 11 --model resnet18 --node-type IFNode --dataset dvsg --step 10 --batch-size 128 --act-fun AtanGrad --device 0```       |
| 1   |  DVS-G  |  LIF+Atan  |   -    |  resnet18   |    128     |    91.29    | ```python main.py --num-classes 11 --model resnet18 --node-type LIFNode --dataset dvsg --step 10 --batch-size 128 --act-fun AtanGrad --device 0```      |
| 1   |  DVS-G  | PLIF+Atan  |   -    |  resnet18   |    128     |    90.15    | ```python main.py --num-classes 11 --model resnet18 --node-type PLIFNode --dataset dvsg --step 10 --batch-size 128 --act-fun AtanGrad --device 0```     |
| 1   |  DVS-G  |  IF+QGate  |   -    | dvs_convnet |    128     |    48.48    | ```python main.py --num-classes 11 --model dvs_convnet --node-type IFNode --dataset dvsg --step 10 --batch-size 128 --act-fun QGateGrad --device 0```   |
| 1   |  DVS-G  | LIF+QGate  |   -    | dvs_convnet |    128     | 92.05/92.42 | ```python main.py --num-classes 11 --model dvs_convnet --node-type LIFNode --dataset dvsg --step 10 --batch-size 128 --act-fun QGateGrad --device 0```  |
| 1   |  DVS-G  | PLIF+QGate |   -    | dvs_convnet |    128     |    91.28    | ```python main.py --num-classes 11 --model dvs_convnet --node-type PLIFNode --dataset dvsg --step 10 --batch-size 128 --act-fun QGateGrad --device 0``` |
| 1   |  DVS-G  |  IF+QGate  |   -    |  resnet18   |    128     |    57.95    | ```python main.py --num-classes 11 --model resnet18 --node-type IFNode --dataset dvsg --step 10 --batch-size 128 --act-fun QGateGrad --device 0```      |
| 1   |  DVS-G  | LIF+QGate  |   -    |  resnet18   |    128     |    90.91    | ```python main.py --num-classes 11 --model resnet18 --node-type LIFNode --dataset dvsg --step 10 --batch-size 128 --act-fun QGateGrad --device 0```     |
| 1   |  DVS-G  | PLIF+QGate |   -    |  resnet18   |    128     |    92.42    | ```python main.py --num-classes 11 --model resnet18 --node-type PLIFNode --dataset dvsg --step 10 --batch-size 128 --act-fun QGateGrad --device 0```    |

### NCALTECH101

| ID  |   Dataset   | Node-type  | Config |    Model    | Batch Size |  Accuracy   | Script                                                                                                                                                          |
|:----|:-----------:|:----------:|:------:|:-----------:|:----------:|:-----------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1   | NCALTECH101 |  IF+QGate  |   -    | dvs_convnet |    128     | 23.09/51.15 | ```python main.py --num-classes 100 --model dvs_convnet --node-type IFNode --dataset NCALTECH101 --step 10 --batch-size 128 --act-fun QGateGrad --device 0```   |
| 1   | NCALTECH101 | LIF+QGate  |   -    | dvs_convnet |    128     | 72.78/75.09 | ```python main.py --num-classes 100 --model dvs_convnet --node-type LIFNode --dataset NCALTECH101 --step 10 --batch-size 128 --act-fun QGateGrad --device 0```  |
| 1   | NCALTECH101 | PLIF+QGate |   -    | dvs_convnet |    128     | 74.61/76.79 | ```python main.py --num-classes 100 --model dvs_convnet --node-type PLIFNode --dataset NCALTECH101 --step 10 --batch-size 128 --act-fun QGateGrad --device 0``` |
| 1   | NCALTECH101 |  IF+QGate  | -/mix  |  resnet18   |    128     | 61.24/60.87 | ```python main.py --num-classes 100 --model resnet18 --node-type IFNode --dataset NCALTECH101 --step 10 --batch-size 128 --act-fun QGateGrad --device 0```      |
| 1   | NCALTECH101 | LIF+QGate  | -/mix  |  resnet18   |    128     | 66.22/70.84 | ```python main.py --num-classes 100 --model resnet18 --node-type LIFNode --dataset NCALTECH101 --step 10 --batch-size 128 --act-fun QGateGrad --device 0```     |
| 1   | NCALTECH101 | PLIF+QGate | -/mix  |  resnet18   |    128     | 69.62/69.87 | ```python main.py --num-classes 100 --model resnet18 --node-type PLIFNode --dataset NCALTECH101 --step 10 --batch-size 128 --act-fun QGateGrad --device 0```    |

Note: 
1. resnet18 is used here by adding a maximum pooling after the initial convolution layer.
However, in the final version of braincog, we remove this pooling layer.
2. mix refers to the use of EventMix as a data augmentation method.
3. We will continue to add other results.


### Citation 
If you find this package helpful, please consider citing it:

```BibTex
@misc{zengbraincogSpikingNeural2022,
  title = {{{braincog}}: {{A Spiking Neural Network}} Based {{Brain-inspired Cognitive Intelligence Engine}} for {{Brain-inspired AI}} and {{Brain Simulation}}},
  shorttitle = {{{braincog}}},
  author = {Zeng, Yi and Zhao, Dongcheng and Zhao, Feifei and Shen, Guobin and Dong, Yiting and Lu, Enmeng and Zhang, Qian and Sun, Yinqian and Liang, Qian and Zhao, Yuxuan and Zhao, Zhuoya and Fang, Hongjian and Wang, Yuwei and Li, Yang and Liu, Xin and Du, Chengcheng and Kong, Qingqun and Ruan, Zizhe and Bi, Weida},
  year = {2022},
  month = jul,
  number = {arXiv:2207.08533},
  eprint = {2207.08533},
  eprinttype = {arxiv},
  primaryclass = {cs},
  publisher = {{arXiv}},
  doi = {10.48550/arXiv.2207.08533}
}
```
