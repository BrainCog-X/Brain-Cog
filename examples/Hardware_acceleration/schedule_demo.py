from pynq import PL
from pynq import Overlay
from pynq import allocate
from pynq import MMIO
import numpy as np
import time
from firefly_utils import *

ol = Overlay('accel/sys_wrapper.bit')

input_channels = 32
output_channels = 64
time_step = 4
kernel_size = 3
parallel_channel = 16
width = 16
height = 16
threshold = 64

input_spike = np.random.randint(0, 2, [time_step, input_channels, height, width], dtype=bool)
weight = np.random.randint(-64, 64, [output_channels, input_channels, kernel_size, kernel_size], dtype=np.int8)
bias = np.random.randint(-64, 64, [output_channels, 1], dtype=np.int16)
out = conv_forward_fast(input_spike, weight, bias)
output_spike = integrate_and_fire(out, threshold)

input_spike_data = convert_input_spike(input_spike, parallel_channel)
weight_data = convert_weight(weight, parallel_channel)
bias_data = bias.flatten()

input_buffer = allocate(shape=input_spike_data.shape, dtype='u1')
weight_buffer = allocate(shape=weight_data.shape, dtype='u1')
bias_buffer = allocate(shape=bias_data.shape, dtype=np.int16)
output_buffer = allocate(shape=(int(output_spike.size / 8),), dtype='u1')
mmio = MMIO(0x0400000000, 0x1000)

input_buffer[:] = input_spike_data
weight_buffer[:] = weight_buffer
bias_buffer[:] = bias_data
input_buffer.flush()
weight_buffer.flush()
bias_buffer.flush()

schedule = FireFlyConvSchedule(
    input_buffer_addr=input_buffer.device_address,
    output_buffer_addr=output_buffer.device_address,
    weight_buffer_addr=weight_buffer.device_address,
    bias_buffer_addr=bias_buffer.device_address,

    parallel_channel=parallel_channel,
    input_channels=input_channels,
    output_channels=output_channels,
    width=width,
    height=height,
    time_step=time_step,
    threshold=threshold
)

schedule.send_config(mmio)

start = time.time()
schedule.begin_schedule(mmio)
end = time.time()

schedule.clear_schedule(mmio)

ms = (end - start) * 10 ** 6
print(f"Elapsed {ms:.03f} micro secs.")

output_buffer.invalidate()
test_output = recover_output_spike(output_buffer, time_step, output_channels, parallel_channel, height, width)

err = (test_output != output_spike).sum()
print(f"Compute Error {err}/{time_step * output_channels * height * width}.")
