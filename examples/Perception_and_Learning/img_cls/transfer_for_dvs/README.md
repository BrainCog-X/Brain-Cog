# Script for all experiments

## Baseline

1. CIFAR10-DVS
```shell 
python main.py --model VGG_SNN --node-type LIFNode --dataset dvsc10 --step 10 --batch-size 120 --act-fun QGateGrad --device 5 --seed 42 --DVS-DA --traindata-ratio 1.0 --smoothing 0.0 --TET-loss-first --TET-loss-second
```

2. N-Caltech 101

```shell
python main.py --model VGG_SNN --node-type LIFNode --dataset NCALTECH101 --step 10 --batch-size 120 --act-fun QGateGrad --device 7 --seed 42 --num-classes 101 --traindata-ratio 1.0 --smoothing 0.0 --TET-loss-first --TET-loss-second
```

3. Omniglot

```shell
python main.py --model SCNN --node-type LIFNode --dataset nomni --step 12 --batch-size 64 --num-classes 1623 --act-fun QGateGrad --epochs 200 --device 6 --log-interval 200 --smoothing 0.0 --seed 42 --lr 0.01 --min-lr 1e-5
```



## Our Method

1. CIFAR10-DVS

```shell
python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset cifar10 --target-dataset dvsc10 --step 10 --batch-size 120 --act-fun QGateGrad --device 1 --seed 42 --traindata-ratio 1.0 --smoothing 0.0 --domain-loss --semantic-loss --DVS-DA --TET-loss-first --TET-loss-second
```

2. N-Caltech 101

```shell
python main_transfer.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset CALTECH101 --target-dataset NCALTECH101 --step 10 --batch-size 120 --act-fun QGateGrad --device 5 --seed 42 --num-classes 101 --traindata-ratio 1.0 --domain-loss --semantic-loss --semantic-loss-coefficient 0.001 --TET-loss-first --TET-loss-second&
```

3. N-Omniglot

```shell
python main_transfer.py --model Transfer_SCNN --node-type LIFNode --source-dataset omni --target-dataset nomni --step 12 --batch-size 64 --num-classes 1623 --act-fun QGateGrad --epochs 200 --device 6 --log-interval 200 --smoothing 0.0 --seed 42 --domain-loss --semantic-loss --semantic-loss-coefficient 0.5 --lr 0.01 --min-lr 1e-5
```



## Visualization Loss-landscape

you should git clone from https://github.com/tomgoldstein/loss-landscape first.

```shell
HDF5_USE_FILE_LOCKING="FALSE" mpirun -n 4 -mca btl ^openib python main_visual_losslandscape.py --model VGG_SNN --node-type LIFNode --source-dataset cifar10 --target-dataset dvsc10 --step 10 --batch-size 1000 --eval --eval_checkpoint /home/TransferLearning_For_DVS/Resultes_new_compare/Baseline/VGG_SNN-dvsc10-10-seed_42-bs_120-DA_True-ls_0.0-traindataratio_0.1-TET_first_False-TET_second_False/last.pth.tar --mpi --x=-1.0:1.0:51 --y=-1.0:1.0:51 --dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn --plot --DVS-DA --smoothing 0.0 --traindata-ratio 0.1
```



```shell
python main_visual_losslandscape.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset CALTECH101 --target-dataset NCALTECH101 --step 10 --batch-size 500 --eval --eval_checkpoint /home/TransferLearning_For_DVS/Results_new_compare/train_TCKA_test/Transfer_VGG_SNN-NCALTECH101-10-bs_120-seed_47-DA_False-ls_0.0-lr_0.005-SNR_0-domainLoss_True-semanticLoss_True-domain_loss_coefficient1.0-semantic_loss_coefficient0.001-traindataratio_0.1-TETfirst_True-TETsecond_True/last.pth.tar --mpi --x=-1.0:1.0:51 --y=-1.0:1.0:51 --dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn --plot --smoothing 0.0 --traindata-ratio 0.1 --num-classes 101 --device 5&
```



## Visualization Grad-cam++

you should git clone from https://github.com/jacobgil/pytorch-grad-cam first.

```shell
python GradCAM_visualization.py --model Transfer_VGG_SNN --node-type LIFNode --source-dataset cifar10 --target-dataset dvsc10 --step 10 --batch-size 1 --act-fun QGateGrad --device 6 --seed 42 --smoothing 0.0 --DVS-DA --eval --eval_checkpoint /home/TransferLearning_For_DVS/Results_lastest/train_TCKA_test/Transfer_VGG_SNN-dvsc10-10-bs_120-seed_42-DA_True-ls_0.0-lr_0.005-SNR_0-domainLoss_True-semanticLoss_True-domain_loss_coefficient1.0-semantic_loss_coefficient0.5-traindataratio_1.0-TETfirst_True-TETsecond_True/model_best.pth.tar
```



## Note: Dataset

In order to work with the source and target domain data, the datastes file is tailored, please use `datasets.py` here to replace and override `braincog/datasets/datasets.py` if you want to run transfer learning in this project.



## Citation

If you find the code and dataset useful in your research, please consider citing:
```
@article{he2023improving,
  title={Improving the Performance of Spiking Neural Networks on Event-based Datasets with Knowledge Transfer},
  author={He, Xiang and Zhao, Dongcheng and Li, Yang and Shen, Guobin and Kong, Qingqun and Zeng, Yi},
  journal={arXiv preprint arXiv:2303.13077},
  year={2023}
}

@misc{https://doi.org/10.48550/arxiv.2207.08533,
  doi = {10.48550/ARXIV.2207.08533},
  url = {https://arxiv.org/abs/2207.08533},
  author = {Zeng, Yi and Zhao, Dongcheng and Zhao, Feifei and Shen, Guobin and Dong, Yiting and Lu, Enmeng and Zhang, Qian and Sun, Yinqian and Liang, Qian and Zhao, Yuxuan and Zhao, Zhuoya and Fang, Hongjian and Wang, Yuwei and Li, Yang and Liu, Xin and Du, Chengcheng and Kong, Qingqun and Ruan, Zizhe and Bi, Weida},
  title = {BrainCog: A Spiking Neural Network based Brain-inspired Cognitive Intelligence Engine for Brain-inspired AI and Brain Simulation},
  publisher = {arXiv},
  year = {2022},
}
```


## Contents

If you are confused about using it or have other feedback and comments, please feel free to contact us via [hexiang2021@ia.ac.cn](hexiang2021@ia.ac.cn).

Have a good day!