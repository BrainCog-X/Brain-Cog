import math

import numpy as np
from einops import rearrange


def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return k, i, j


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)
    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def max_pool_forward_reshape(x, pool_param):
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']
    assert pool_height == pool_width == stride, 'Invalid pool params'
    assert H % pool_height == 0
    assert W % pool_height == 0
    x_reshaped = x.reshape(N, C, int(H / pool_height), pool_height, int(W / pool_width), pool_width)
    out = x_reshaped.max(axis=3).max(axis=4)
    return out


def max_pool_forward_fast(x, pool_param):
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']

    same_size = pool_height == pool_width == stride
    tiles = H % pool_height == 0 and W % pool_width == 0
    if same_size and tiles:
        out = max_pool_forward_reshape(x, pool_param)
    else:
        out = max_pool_forward_im2col(x, pool_param)
    return out


def max_pool_forward_im2col(x, pool_param):
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']

    assert (H - pool_height) % stride == 0, 'Invalid height'
    assert (W - pool_width) % stride == 0, 'Invalid width'

    out_height = int((H - pool_height) / stride + 1)
    out_width = int((W - pool_width) / stride + 1)

    x_split = x.reshape(N * C, 1, H, W)
    x_cols = im2col_indices(x_split, pool_height, pool_width, padding=0, stride=stride)
    x_cols_argmax = np.argmax(x_cols, axis=0)
    x_cols_max = x_cols[x_cols_argmax, np.arange(x_cols.shape[1])]
    out = x_cols_max.reshape(out_height, out_width, N, C).transpose(2, 3, 0, 1)
    return out


def conv_forward_fast(x, w, b, pad=1, stride=1):
    N, C, H, W = x.shape
    # x = x.astype(np.int32)
    w = w.astype(np.int32)
    b = b.astype(np.int16)
    num_filters, _, filter_height, filter_width = w.shape

    out_height = int((H + 2 * pad - filter_height) / stride + 1)
    out_width = int((W + 2 * pad - filter_width) / stride + 1)
    out = np.zeros((N, num_filters, out_height, out_width), dtype=np.int32)

    x_cols = im2col_indices(x, w.shape[2], w.shape[3], pad, stride)
    res = w.reshape((w.shape[0], -1)).dot(x_cols) + b.reshape(-1, 1)
    out = res.reshape(w.shape[0], out.shape[2], out.shape[3], x.shape[0])
    out = out.transpose(3, 0, 1, 2)

    return out


def spike_map_pack_to_bytes_array(spike_map, parallel):
    buf_in = rearrange(spike_map, 't (c p) h w->t c h w p', p=parallel)
    buf_in = np.packbits(buf_in.flatten(), bitorder='little')
    return buf_in


def bytes_array_split_to_spike_map(buf_in, time_step, parallel, H, W):
    unpacked = np.unpackbits(buf_in, bitorder='little')
    unpacked = rearrange(unpacked, '(t c h w p)->t (c p) h w', t=time_step, p=parallel, h=H, w=W)
    return unpacked


def preprocess(model_config_list, x, parallel):
    time_step = model_config_list[1]["time_step"]
    scale = model_config_list[0]["scale"]
    zero_point = model_config_list[0]["zero_point"]
    weight = model_config_list[1]["weight"]
    bias = model_config_list[1]["bias"]
    threshold = model_config_list[1]["threshold"]

    encode_in = np_quantize_prepare(x, scale, zero_point)
    encode_in = np.expand_dims(encode_in, axis=0)
    firefly_in = direct_coding(encode_in, weight, bias, time_step, threshold)
    packed = spike_map_pack_to_bytes_array(firefly_in, parallel)
    return firefly_in, packed


def integrate_and_fire(y, threshold):
    membrane = np.zeros(y.shape[1:], dtype=np.int32)
    out_spike = []
    for v in y:
        membrane = membrane + v
        o = membrane > threshold
        out_spike.append(o)
        membrane[o] = 0
    return np.array(out_spike)


def direct_coding(x, w, b, time_step, threshold):
    x = x.repeat(time_step, axis=0)
    out_spike = conv_ifnode_forward(x, w, b, threshold)
    return out_spike


def conv_ifnode_forward(x, w, b, threshold):
    y = conv_forward_fast(x, w, b)
    out_spike = integrate_and_fire(y, threshold)
    return out_spike


def conv_ifnode_maxpool_forward(x, w, b, threshold):
    y = conv_ifnode_forward(x, w, b, threshold)
    out_spike = max_pool_forward_fast(y, {'pool_height': 2, 'pool_width': 2, 'stride': 2})
    return out_spike


def linear_wta_forward(x, w, b):
    x = x.astype(np.int32)
    w = w.astype(np.int32)
    b = b.astype(np.int32)
    x = x.reshape([x.shape[0], -1])
    x = np.pad(x, ((0, 0), (0, w.shape[1] - x.shape[1])), 'constant')
    out = np.dot(x, w.T) + b
    out_sum = out.sum(axis=0)
    max_index = out_sum.argmax()
    return out, max_index


def linear_ifnode_forward(x, w, b, threshold):
    x = x.astype(np.int32)
    w = w.astype(np.int32)
    b = b.astype(np.int32)
    x = x.reshape([x.shape[0], -1])
    x = np.pad(x, ((0, 0), (0, w.shape[1] - x.shape[1])), 'constant')
    out = np.dot(x, w.T) + b
    out_spike = integrate_and_fire(out, threshold)
    return out_spike


def pad_conv_weight_round_to_parallel(parallel, weight, pad_output_channel_only=False):
    output_channel = weight.shape[0]
    input_channel = weight.shape[1]
    padded_output_channel = (parallel - (output_channel % parallel)) % parallel
    padded_input_channel = (parallel - (input_channel % parallel)) % parallel
    if pad_output_channel_only:
        padded_input_channel = 0
    new_weight = np.pad(weight, ((0, padded_output_channel), (0, padded_input_channel), (0, 0), (0, 0)), 'constant')
    return new_weight


def pad_linear_weight_round_to_parallel(parallel, weight):
    output_channel = weight.shape[0]
    padded_output_channel = (parallel - (output_channel % parallel)) % parallel
    new_weight = np.pad(weight, ((0, padded_output_channel), (0, 0)), 'constant')
    return new_weight


def pad_linear_weight_round_to_factor(weight, factor):
    input_channel = weight.shape[1]
    round_channel = int(math.ceil(input_channel / factor) * factor)
    padded_input_channel = round_channel - input_channel
    new_weight = np.pad(weight, ((0, 0), (0, padded_input_channel)), 'constant')
    return new_weight


def pad_bias_round_to_parallel(parallel, bias, pad_value=0):
    channel = bias.shape[0]
    padded_channel = (parallel - (channel % parallel)) % parallel
    new_bias = np.pad(bias, (0, padded_channel), 'constant', constant_values=pad_value)
    return new_bias


def np_quantize_per_tensor(x, scale, zero_point):
    q_min = np.iinfo(np.int8).min
    q_max = np.iinfo(np.int8).max
    x = np.round(x / scale + zero_point)
    x = np.clip(x, q_min, q_max)
    return x.astype(np.int8)


def np_quantize_prepare(x, scale, zero_point):
    x = np_quantize_per_tensor(x, scale, zero_point)
    return x - zero_point


def conv_weight_channel_tiling(parallel, weight):
    return rearrange(weight, '(o op) (i ip) kr kc -> (o i kr kc) ip op', op=parallel, ip=parallel)


def linear_weight_channel_tiling(parallel, weight):
    return rearrange(weight, '(o op) (i ip) -> (o i) ip op', op=parallel, ip=parallel)


def conv_to_linear_weight_tiling(parallel, h, w, weight):
    rearrange(weight, '(o op) (i ip h w)-> (o i h w) ip op', op=parallel, ip=parallel, h=h, w=w)


def init_input_buffer(input_spikes,
                      parallel=16,
                      stride_of_channel=8 * 1024,
                      stride_of_time_step=512 * 1024):
    t, c, h, w = input_spikes.shape
    input_spikes_rearrange = rearrange(input_spikes, 't (c p) h w -> t c (h w p)', p=parallel)
    pack_spikes = np.packbits(input_spikes_rearrange, axis=-1, bitorder='little')
    input_buffer = np.zeros(stride_of_time_step * t, dtype=np.uint8)
    length = int(h * w * parallel / 8)
    for i in range(t):
        for j in range(int(c / parallel)):
            addr = i * stride_of_time_step + j * stride_of_channel
            input_buffer[addr:addr + length] = pack_spikes[i, j]
    return input_buffer


def get_from_output_buffer(output_buffer,
                           t, c, h, w,
                           parallel=16,
                           stride_of_channel=8 * 1024,
                           stride_of_time_step=512 * 1024):
    ret = []
    length = int(h * w * parallel / 8)
    for i in range(t):
        for j in range(int(c / parallel)):
            addr = i * stride_of_time_step + j * stride_of_channel
            ret.append(output_buffer[addr:addr + length])
    ret = np.array(ret)
    ret = np.unpackbits(ret, axis=-1, bitorder='little')
    ret = rearrange(ret, '(t c) (h w p) -> t (c p) h w', p=parallel, t=t, h=h, w=w)
    return ret.astype(bool)


def get_output_index(buffer, parallel):
    valid_data = buffer[:parallel]
    if parallel == 16:
        return np.unpackbits(valid_data[12:14], bitorder='little').argmax()
    elif parallel == 32:
        return np.unpackbits(valid_data[24:28], bitorder='little').argmax()
    else:
        return 0


def save_model_config_list(model_config_list, path):
    np.save(path, model_config_list)
    return


def load_model_config_list(path):
    model_config_list = np.load(path, allow_pickle=True)
    return model_config_list
