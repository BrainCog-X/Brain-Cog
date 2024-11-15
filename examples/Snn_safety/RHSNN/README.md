* To train a SNN with AT on CIFAR-10:
```
python train.py --adv_training --attack_iters 1 --epsilon 4 --alpha 4 --network ResNet18 --batch_size 64 --worker 4 --node_type LIF --save_dir AT --device cuda:1 --time_step 8 --dataset cifar10
```

* To train a RHSNN with AT on CIFAR-10:
```
python train.py --adv_training --attack_iters 1 --epsilon 4 --alpha 4 --network ResNet18 --batch_size 64 --worker 4 --node_type RHLIF --save_dir AT_RH_1 --device cuda:1 --time_step 8 --dataset cifar10
```

* To train a RHSNN with RAT on CIFAR-10:
```
python train.py --adv_training --attack_iters 1 --epsilon 4 --alpha 4 --network ResNet18 --batch_size 64 --worker 4 --node_type RHLIF --save_dir RAT_RH_1 --device cuda:1 --parseval --beta 0.004 --time_step 8 --dataset cifar10
```

* To train a RHSNN with SR on CIFAR-10:
```
python train.py --adv_training --attack_iters 1 --epsilon 4 --alpha 4 --network ResNet18 --batch_size 64 --worker 4 --node_type RHLIF --save_dir SR_RH_1 --device cuda:1 --SR --time_step 8 --dataset cifar10
```

* To evaluate the performance of RHSNN on CIFAR-10:

```
python evaluate.py --network ResNet18 --attack_type all --batch_size 32 --worker 4 --node_type RHLIF --pretrain RAT_RH_1/weight_r.pth --save_dir RAT --device cuda:1 --time_step 8 --dataset cifar10
```

