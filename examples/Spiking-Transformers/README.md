# Spiking Transformers Reproduced With Braincog
Here is the current Spiking Transformer code reproduced using [BrainCog](http://www.brain-cog.network/). Welcome to follow the work of BrainCog and utilize the [BrainCog framework](https://github.com/BrainCog-X/Brain-Cog) to create relevant brain-inspired AI endeavors. The works implemented here will also be merged into BrainCog Repo.

### Models
**Spikformer(ICLR 2023)**
[Zhou, Z., Zhu, Y., He, C., Wang, Y., Yan, S., Tian, Y., & Yuan, L. (2022). Spikformer: When spiking neural network meets transformer. arXiv preprint arXiv:2209.15425.](https://openreview.net/forum?id=frE4fUwz_h)
![alt text](/img/spikformer.png)

**Spike-driven Transformer(Nips 2023)**
[Yao, M., Hu, J., Zhou, Z., Yuan, L., Tian, Y., Xu, B., & Li, G. (2024). Spike-driven transformer. Advances in Neural Information Processing Systems, 36.](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ca0f5358dbadda74b3049711887e9ead-Abstract-Conference.html)
![alt text](/img/sdv1.png)


**Spike-driven Transformer V2(ICLR 2024)**
[Yao, M., Hu, J., Hu, T., Xu, Y., Zhou, Z., Tian, Y., ... & Li, G. (2023, October). Spike-driven Transformer V2: Meta Spiking Neural Network Architecture Inspiring the Design of Next-generation Neuromorphic Chips. In The Twelfth International Conference on Learning Representations.](https://openreview.net/forum?id=1SIBN5Xyw7)
![alt text](/img/sdv2.png)

## Models in comming soon
**SpikingResFormer**
[Shi, X., Hao, Z., & Yu, Z. (2024). SpikingResformer: Bridging ResNet and Vision Transformer in Spiking Neural Networks. arXiv preprint arXiv:2403.14302.](https://arxiv.org/abs/2403.14302)

**TIM**
[Shen, S., Zhao, D., Shen, G., & Zeng, Y. (2024). TIM: An Efficient Temporal Interaction Module for Spiking Transformer. arXiv preprint arXiv:2401.11687.](https://arxiv.org/abs/2401.11687)

**SGLFormer(Frontiers in Neuroscience)**
[Zhang, H., Zhou, C., Yu, L., Huang, L., Ma, Z., Fan, X., ... & Tian, Y. (2024). SGLFormer: Spiking Global-Local-Fusion Transformer with High Performance. Frontiers in Neuroscience, 18, 1371290.](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1371290/full)

**QKFormer(CVPR2024)**
[Zhou, C., Zhang, H., Zhou, Z., Yu, L., Huang, L., Fan, X., ... & Tian, Y. (2024). QKFormer: Hierarchical Spiking Transformer using QK Attention. arXiv preprint arXiv:2403.16552.](https://arxiv.org/abs/2403.16552)


## Requirments
- Braincog
- einops >= 0.4.1
- timm >= 0.5.4

## Training Examples
### Training on CIFAR10-DVS
python main.py --dataset dvsc10 --epochs 500 --batch-size 16 --seed 42 --event-size 64 --model spikformer_dvs
### Training on ImageNet
python main.py --dataset imnet --epochs 500 --batch-size 16 --seed 42 --model spikformer