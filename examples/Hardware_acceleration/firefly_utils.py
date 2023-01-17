import numpy as np


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


def conv_forward_fast(x, w, b, pad=1, stride=1):
    N, C, H, W = x.shape
    num_filters, _, filter_height, filter_width = w.shape

    out_height = int((H + 2 * pad - filter_height) / stride + 1)
    out_width = int((W + 2 * pad - filter_width) / stride + 1)
    out = np.zeros((N, num_filters, out_height, out_width), dtype=x.dtype)

    x_cols = im2col_indices(x, w.shape[2], w.shape[3], pad, stride)
    res = w.reshape((w.shape[0], -1)).dot(x_cols) + b.reshape(-1, 1)
    out = res.reshape(w.shape[0], out.shape[2], out.shape[3], x.shape[0])
    out = out.transpose(3, 0, 1, 2)

    return out


def integrate_and_fire(y, threshold):
    membrane = np.zeros([y.shape[1], y.shape[2], y.shape[3]], dtype=np.int32)
    out_spike = []
    for v in y:
        membrane = membrane + v
        o = membrane > threshold
        out_spike.append(o)
        membrane[o] = 0
    return np.array(out_spike)


def convert_input_spike(x, parallel_channel):
    t, ic, h, w = x.shape
    x = x.reshape([t, int(ic / parallel_channel), parallel_channel, h, w])
    x = x.transpose(0, 1, 3, 4, 2).flatten()
    return np.packbits(x, bitorder='little')


def convert_weight(w, parallel_channel):
    oc, ic, ky, kx = w.shape
    w = w.reshape([int(oc / parallel_channel), parallel_channel, int(ic / parallel_channel), parallel_channel, -1])
    w = w.transpose(0, 2, 4, 3, 1).flatten()
    return w


def recover_output_spike(o, time_step, out_channels, parallel_channel, height, width):
    out_bits = np.unpackbits(o, bitorder='little').astype(bool)
    out_bits = out_bits.reshape([int(out_channels / parallel_channel), time_step, height, width, parallel_channel])
    out_bits = out_bits.transpose(1, 0, 4, 2, 3).reshape([time_step, out_channels, height, width])
    return out_bits


class FireFlyConvSchedule:
    def __init__(
            self,
            input_buffer_addr,
            output_buffer_addr,
            weight_buffer_addr,
            bias_buffer_addr,
            parallel_channel=16,
            kernel_size=3,
            input_channels=64,
            output_channels=128,
            width=32,
            height=32,
            enable_pooling=False,
            time_step=8,
            threshold=64
    ):
        self.ctrl_addr_os = 0
        self.input_addr_os = self.ctrl_addr_os + 0x100
        self.param_addr_os = self.ctrl_addr_os + 0x200
        self.output_addr_os = self.ctrl_addr_os + 0x300

        self.input_buffer_addr = np.uint32(input_buffer_addr)
        self.output_buffer_addr = np.uint32(output_buffer_addr)
        self.weight_buffer_addr = np.uint32(weight_buffer_addr)
        self.bias_buffer_addr = np.uint32(bias_buffer_addr)

        self.parallel_channel = np.uint32(parallel_channel)
        self.kernel_size = np.uint32(kernel_size)
        self.input_channels = np.uint32(input_channels)
        self.output_channels = np.uint32(output_channels)
        self.width = np.uint32(width)
        self.height = np.uint32(height)
        self.enable_pooling = np.uint32(enable_pooling)
        self.time_step = np.uint32(time_step)
        self.threshold = np.uint32(threshold)

        self.numOfIFMs = np.uint32(input_channels / parallel_channel - 1)
        self.numOfOFMs = np.uint32(output_channels / parallel_channel - 1)
        self.numOfTimeSteps = np.uint32(time_step - 1)
        self.numOfTimeStepIFMs = np.uint32((input_channels / parallel_channel) * time_step - 1)
        self.numOfTimeStepOFMs = np.uint32((output_channels / parallel_channel) * time_step - 1)
        self.weightsLengh = np.uint32(kernel_size * kernel_size * input_channels - 1)

        self.configReg0 = (self.enable_pooling << 48) + (self.numOfTimeStepOFMs << 36) + (
                self.numOfTimeStepIFMs << 24) + (self.numOfTimeSteps << 16) + (self.numOfOFMs << 8) + self.numOfIFMs
        self.configReg1 = (self.weightsLengh << 32) + ((self.height - 1) << 16) + self.width - 1
        self.configReg2 = self.threshold

        self.total_pass = np.uint32(self.output_channels / self.parallel_channel)
        self.bias_len = self.output_channels * 2
        self.weight_len = self.output_channels * self.input_channels * self.kernel_size * self.kernel_size
        self.in_len_per_batch = np.uint32(self.time_step * self.input_channels * self.height * self.width / 8)
        if enable_pooling:
            self.out_len_per_batch = np.uint32(self.time_step * self.output_channels * self.height * self.width / 8)
        else:
            self.out_len_per_batch = np.uint32(self.time_step * self.output_channels * self.height * self.width / 8 / 4)

        self.bias_cmd = (self.bias_buffer_addr << 32) + (1 << 23) + self.bias_len
        self.weight_cmd = (self.weight_buffer_addr << 32) + (1 << 23) + self.weight_len
        self.out_cmd = (self.output_buffer_addr << 32) + (1 << 23) + self.out_len_per_batch
        self.in_cmd = (self.input_buffer_addr << 32) + (1 << 23) + self.in_len_per_batch

    def send_config(self, ctrl):
        ctrl.write(self.ctrl_addr_os, self.configReg0.tobytes())
        ctrl.write(self.ctrl_addr_os + 0x08, self.configReg1.tobytes())
        ctrl.write(self.ctrl_addr_os + 0x10, self.configReg2.tobytes())

    def begin_schedule(self, ctrl):
        print("send begin")
        ctrl.write(self.param_addr_os, self.bias_cmd.tobytes())
        ctrl.write(self.param_addr_os, self.weight_cmd.tobytes())
        ctrl.write(self.output_addr_os, self.out_cmd.tobytes())
        cnt = 0
        while cnt < self.total_pass:
            if ctrl.read(self.input_addr_os, length=8) == 0:
                ctrl.write(self.input_addr_os, self.in_cmd.tobytes())
                print("-")
                cnt = cnt + 1
        print("send finish")
        while ctrl.read(self.ctrl_addr_os + 0x18, length=8) == 0:
            pass

    def clear_schedule(self, ctrl):
        print("before clear", ctrl.read(self.ctrl_addr_os + 0x18, length=8))
        ctrl.write(self.ctrl_addr_os + 0x18, 0)
        print("after clear", ctrl.read(self.ctrl_addr_os + 0x18, length=8))
