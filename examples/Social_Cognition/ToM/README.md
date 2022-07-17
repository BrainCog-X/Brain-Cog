# Requirments

* numpy
* scipy
* pytorch >= 1.7.0
* torchvision
* pygame

# Run
## Train 
* the file to be run: main_both.py 
* args:
    * the path to save net_NPC: --save_net_N
    * the path to save net_a: --save_net_a
    * time steps: --T

```bash
python main_both.py --save_net_N=net_NPC.pth --save_net_a=net_agent.pth --episodes=45 --trajectories=30 --T=50 --mode=train --task=both
```

## Test
You can use the weigts saved by taining in the test environment.

```bash
python main_ToM.py --save_net_N=net_NPC.pth --save_net_a=net_agent.pth --episodes=45 --trajectories=30 --T=50 --mode=train --task=both
```

