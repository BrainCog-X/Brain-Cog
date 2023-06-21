## FireFly: A High-Throughput Hardware Accelerator for Spiking Neural Networks

### Demo of Deploying SNNs on FPGA platform

This is an example of deploying an SNN model on Xilinx Zynq Ultrascale FPGA based on Braincog.

### Requirements

- Xilinx Zynq Ultrascale FPGA evaluation board Ultra96v2 or ZCU104.
- PYNQ images for the chosen evaluation boards. You can download the latest pre-compiled images from the [PYNQ website](http://www.pynq.io/board.html), or you can compile a new one following the [PYNQ Tutorial](https://pynq.readthedocs.io/en/latest/). Install the PYNQ image to the SD card, and boot the evaluation board in SD mode.

### Examples

Clone the project to fetch the necessary bitstream files and pre-processed SNN models, copy all the files to the Ultra96v2 or ZCU104 board.

```shell
git clone https://github.com/adamgallas/firefly_v1_cifar_test
```

Open a terminal in Ultra96v2 or ZCU104. Install einops on Ultra96v2 or ZCU104.

```shell
cd firefly_v1_common
pip install einops-0.6.0-py3-none-any.whl
```

Run CIFAR10  classification test on Ultra96v2:

```shell
python ultra96_test.py
```

Run CIFAR10  classification test on ZCU104:

```python
python zcu104_test.py
```

### Citation

### Citation 
If you find this work helpful, please consider citing it:

```BibTex
@article{li2023firefly,
  title={FireFly: A High-Throughput Hardware Accelerator for Spiking Neural Networks With Efficient DSP and Memory Optimization},
  author={Li, Jindong and Shen, Guobin and Zhao, Dongcheng and Zhang, Qian and Zeng, Yi},
  journal={IEEE Transactions on Very Large Scale Integration (VLSI) Systems},
  year={2023},
  publisher={IEEE}
}
```