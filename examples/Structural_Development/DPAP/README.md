# Similarity-based context aware continual learning for spiking neural networks #

## Requirments ##
* albumentations==1.1.0
* easydict==1.9
* matplotlib==3.5.1
* nni==2.10
* numpy==1.22.4
* opencv_python==4.5.5.62
* Pillow==9.3.0
* sacred==0.8.2
* scikit_learn==1.1.3
* scipy==1.9.3
* tensorboardX==2.5.1
* thop==0.0.31.post2005241907
* torch==1.8.1+cu111
* torchvision==0.9.1+cu111


## Run ##

``` CUDA_VISIBLE_DEVICES=0 python3 -m main train with "./SCA-SNN/configs/train.yaml" exp.name="cifar_b0_10s" exp.savedir="./log/" exp.saveckpt="./ckpts_cifar_b0_10s/" exp.ckptdir="./log/" exp.tensorboard_dir="./tensorboard/" exp.debug=False --name="cifar_b0_10s" -D --force```

## Citation ##
If you find the code and dataset useful in your research, please consider citing:
```
@article{han2024similarity,
  title={Similarity-based context aware continual learning for spiking neural networks},
  author={Han, Bing and Zhao, Feifei and Li Yang and Kong Qingqun and Li Xianqi and Zeng, Yi},
  year={2024}
  }
  
@article{zeng2023braincog,
  title={Braincog: A spiking neural network based, brain-inspired cognitive intelligence engine for brain-inspired ai and brain simulation},
  author={Zeng, Yi and Zhao, Dongcheng and Zhao, Feifei and Shen, Guobin and Dong, Yiting and Lu, Enmeng and Zhang, Qian and Sun, Yinqian and Liang, Qian and Zhao, Yuxuan and others},
  journal={Patterns},
  volume={4},
  number={8},
  year={2023},
  publisher={Elsevier},
}
```

Enjoy!
