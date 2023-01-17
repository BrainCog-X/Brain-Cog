# Demo of implementing SNNs on FPGA platform
This is an example of implementing SNNs on Xilinx Zynq Ultrascale FPGA based on Braincog.

## Requirement

- A Xilinx Zynq Ultrascale FPGA evaluation board, such as Ultra96v2, KV260, KR260, ZCU104, ZCU102 or other third-party evaluation boards. Currently xczu3eg is the smallest device we support.
- PYNQ images for the chosen evaluation boards. You can download the latest pre-compiled images from the [PYNQ website](http://www.pynq.io/board.html), or you can compile a new one following the [PYNQ Tutorial](https://pynq.readthedocs.io/en/latest/).
- You don't need expert knowledge on FPGA or hardware design, but at least you need to read the [PYNQ documentation](https://pynq.readthedocs.io/en/latest/) first.

## Supported Layers

- Currently, we only support 3x3 convolution, 2x2 max-pooling and fully-connected layer.
- Currently, we only support Integrate-and-Fire neurons.

## Demo of SNNs Preprocessing

The SNN models need several steps of preprocessing before FPGA deployment.

In the state-of-the-art SNN models, a convolution layers are often followed by a batch-normalization layer. The BN layer can be fused to the convolution layer in advance to reduce computational complexity.

In hardware design, floating point caculations are extremely expensive. Quantization methods need to be applied to reduce resource consumption.

[fuse_and_quant_demo.py](fuse_and_quant_demo.py) shows how to fuse and quantize a trained SNN network and generate the files needed for further deployment.

A trained SNN model can be downloaded from [here](https://1drv.ms/u/s!Amgjsr1eCIFlgSVJHJH6ZHFg_4jg?e=8pfdjb).

## Demo of SNN Tasks Scheduling

In our hardware architecture, the inference process of a SNN model is arranged in a layer-wise fashion. The inference process of a single layer is folded into several phases.

[schedule_demo.py](schedule_demo.py) shows how to schedule a single SNN layer inference process on the Zynq device.

We adopt the PYNQ framework to make life easy. The hardware bitstream are already generated in advance, you don't need to be an FPGA expert to run this demo. The bitstream is loaded to the FPGA by the python script above.

The hardware configurations needed can be downloaded from [here](https://1drv.ms/u/s!Amgjsr1eCIFlgSQvOf-sMswiO27i?e=YHpdre). Copy them to the PYNQ directory 

### Citation 
If you find this work helpful, please consider citing it:

```BibTex
@article{li2023firefly,
  title={FireFly: A High-Throughput and Reconfigurable Hardware Accelerator for Spiking Neural Networks},
  author={Li, Jindong and Shen, Guobin and Zhao, Dongcheng and Zhang, Qian and Yi, Zeng},
  journal={arXiv preprint arXiv:2301.01905},
  year={2023}
}
```
