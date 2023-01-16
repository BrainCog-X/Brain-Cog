import torch
from torch import nn
from functools import partial
from timm.models import register_model
from braincog.base.node.node import *
from braincog.base.encoder.encoder import *
from braincog.model_zoo.vgg_snn import *
from braincog.model_zoo.base_module import BaseModule, BaseConvModule
from braincog.datasets import is_dvs_data
from braincog.datasets.datasets import *


def fuse_conv_and_bn(conv, bn):
    #
    # init
    fusedconv = torch.nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=True
    )
    #
    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))
    #
    # prepare spatial bias
    if conv.bias is not None:
        b_conv = conv.bias
    else:
        b_conv = torch.zeros(conv.weight.size(0))
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.matmul(w_bn, b_conv) + b_bn)
    #
    # we're done
    return fusedconv


def get_quantize_para_int(para_in):
    max_val = para_in.max()
    min_val = para_in.min()
    scale = (max_val - min_val) / 256
    zp = torch.tensor(0.0).cuda()
    wq = torch.quantize_per_tensor(para_in, scale, zp, torch.qint8)
    return scale, zp, wq.int_repr().float()


checkpoint = torch.load("model/snn7_tiny.tar")
model = SNN7_tiny(dataset='cifar10', layer_by_layer=True, step=4, node_type=IFNode, threshold=.5)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
batchSize = 1
_, test_loader, _, _ = get_cifar10_data(batch_size=batchSize)
torch.set_grad_enabled(False)

model_fused = nn.Sequential()
model_fused.append(model.encoder)

for layer in model.feature:
    with torch.no_grad():
        if isinstance(layer, BaseConvModule):
            model_fused.append(fuse_conv_and_bn(layer.conv, layer.bn))
            model_fused.append(layer.node)
        if isinstance(layer, nn.MaxPool2d):
            model_fused.append(layer)

for layer in model.fc:
    model_fused.append(layer)

input_max = torch.tensor(2.65).cuda()
input_min = torch.tensor(-2.12).cuda()
quant_max = 256
input_scale = (input_max - input_min) / quant_max
input_zp = (quant_max / 2 - 1 - input_max / input_scale).round()
threshold = torch.tensor(0.5).cuda()
model_fused[2].set_n_threshold(torch.tensor(1200))

encode_scale, encode_zp, quantized_weight = get_quantize_para_int(model_fused[1].weight.cuda())
quantized_bias = (
    torch.quantize_per_tensor(model_fused[1].bias.cuda(), encode_scale, encode_zp, torch.qint32)).int_repr().float()
model_fused[1].weight.copy_(quantized_weight)
model_fused[1].bias.copy_(quantized_bias)
threshold_int = (torch.quantize_per_tensor(threshold, input_scale * encode_scale, torch.tensor(0).cuda(),
                                           torch.qint32)).int_repr().float()
print(input_scale, input_zp, threshold_int)

last_scale = torch.tensor(0.0).cuda()
last_zp = torch.tensor(0.0).cuda()


for layer in model_fused[3:]:
    with torch.no_grad():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            scale, zp, quantized_weight = get_quantize_para_int(layer.weight.cuda())
            quantized_bias = (torch.quantize_per_tensor(layer.bias.cuda(), scale, zp, torch.qint32)).int_repr().float()
            layer.weight.copy_(quantized_weight)
            layer.bias.copy_(quantized_bias)
            last_scale = scale
            last_zp = zp
        if isinstance(layer, IFNode):
            threshold_int = (torch.quantize_per_tensor(threshold, last_scale.cuda(), last_zp.cuda(),
                                                       torch.qint32)).int_repr().float()
            layer.set_n_threshold(threshold_int)

it = iter(test_loader)
img, label = it.next()
x = (torch.quantize_per_tensor(img.cuda(), input_scale, input_zp.cuda(),
                               torch.qint8)).int_repr().float() - input_zp.cuda()
parallel_channel = 16
conv_index = 0
model_fused.cuda()
for layer in model_fused:
    if hasattr(layer, 'n_reset'):
        layer.n_reset()
    if isinstance(layer, nn.Conv2d):
        if layer.weight.shape[1] % parallel_channel == 0:
            conv_index = conv_index + 1
            print(conv_index, "conv", layer.weight.shape, x.shape)
            os.makedirs("conv_%d" % conv_index, exist_ok=True)

            act = rearrange(x.cpu().numpy(), 't (b p) h w -> t b (h w p)', p=parallel_channel)
            act_file = np.packbits(act.astype(bool).flatten(), bitorder='little')
            weight_file = rearrange(layer.weight, '(o op) (i ip) kr kc -> o i (kr kc) ip op', op=parallel_channel,
                                    ip=parallel_channel).cpu().numpy().astype(np.int8)
            bias_file = layer.bias.cpu().numpy().astype(np.int16)
            act_file.tofile("conv_%d/convin.bin" % conv_index)
            weight_file.tofile("conv_%d/weight.bin" % conv_index)
            bias_file.tofile("conv_%d/bias.bin" % conv_index)
        x = layer(x)
    elif isinstance(layer, IFNode) and conv_index != 0:
        print(conv_index, "if", layer.threshold.data, x.shape)
        x = layer(x)
        act = rearrange(x.cpu().numpy(), 't (b p) h w -> t b (h w p)', p=parallel_channel)
        act_file = np.packbits(act.astype(bool).flatten(), bitorder='little')
        act_file.tofile("conv_%d/ifout.bin" % conv_index)
    elif isinstance(layer, nn.MaxPool2d):
        print(conv_index, "pool", x.shape)
        x = layer(x)
        act = rearrange(x.cpu().numpy(), 't (b p) h w -> t b (h w p)', p=parallel_channel)
        act_file = np.packbits(act.astype(bool).flatten(), bitorder='little')
        act_file.tofile("conv_%d/poolout.bin" % conv_index)
    else:
        x = layer(x)
