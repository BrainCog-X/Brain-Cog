## Requirements:
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
* tonic (special)
* pandas >= 1.4.2  
* xlrd == 1.2.0
## Install

```
# optional, if use datasets 
git clone http://172.18.116.130:3000/floyed/tonic.git
cd tonic 
pip install -e .

pip install -r requirements.txt

git clone http://172.18.116.130:3000/floyed/BrainCog.git
cd BrainCog
pip install -e .
```

## Example:

```python
cd ./examples 
python main.py --model cifar_convnet --dataset cifar10 --node-type PLIFNode --step 8 --device 1
```

## Result

|    Dataset    |      Accuracy       |                                                        Script                                                         |
|:-------------:|:-------------------:|:---------------------------------------------------------------------------------------------------------------------:|
|    CIFAR10    |  94.89 (epoch 535)  |                  ```python main.py --model cifar_convnet --node-type PLIFNode --step 8 --device 5```                  |
|  DVS-CIFAR10  |  80.85 (epoch 343)  |  ```python main.py --device 0 --dataset dvsc10 --node-type PLIFNode --step 10 --model dvs_convnet --batch-size 32```  |

## Log:

### 2022.4.10

* 修改名称为```BrainCog```;
* 添加了文档系统;
* 为```node.py```编写了文档;
* 重新整理了```base```, 分成了五个模块;
* 在130服务器上建立了git仓库;

### 2020.4.29 
* 添加地简化版本的图像分类训练脚本 ```train_simplified.py```.
* 添加了 SNN ```layer wise``` 的前向传播方法.
* 添加了针对事件数据的 ```EventMix``` 方法.
* 修复了一些已知的BUG.