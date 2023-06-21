from standalone_utils import *
from firefly_v1_schedule_on_pynq import *
from pynq import PL
from pynq import Overlay
from pynq import allocate
from pynq import MMIO
import numpy as np
import time
from einops import rearrange
ol = Overlay('firefly_v1_ultra96_bitstream/sys_wrapper.bit')

image = np.load("firefly_v1_cifar10_data/image.npy")
target = np.load("firefly_v1_cifar10_data/target.npy")
model_config_list = load_model_config_list("firefly_v1_cifar10_data/snn7_cifar10_x16.npy")

input_buffer = allocate(shape=(1<<23), dtype=np.uint8)
output_buffer = allocate(shape=(1<<23), dtype=np.uint8)
ctrl_io = MMIO(0x0400000000, 0x400)

schedule_list,output_addr=create_schedule(
    model_config_list=model_config_list,
    ctrl_io=ctrl_io,
    allocate_method=allocate,
    buffer_0=input_buffer,
    buffer_1=output_buffer,
    image_height=32,
    image_width=32,
    time_step=4,
    parallel_channel=16
)

sche, ct_ptr, sche_len = init_firefly_c_lib_with_time("firefly_v1_common/firefly_v1_lib.so", schedule_list)

print(" ----------------- initialize finish!")


print(" ----------------- python schedule begin")
err_cnt = 0
for i in range(len(image)):
    image_0 = image[i]
    target_0 = target[i]
    print(i)

    start = time.time()
    firefly_in, input_packed = preprocess(model_config_list, image_0, 16)
    input_buffer[:input_packed.size]=input_packed
    input_buffer.flush()
    end = time.time()

    elapsed = round((end - start) * 1000000)
    # print("preprocess:", elapsed, "us")

    input_buffer[:input_packed.size]=input_packed
    input_buffer.flush()
    start = time.time()
    schedule_run_all(schedule_list)
    end = time.time()

    elapsed = round((end - start) * 1000000)
    print("snn inference:", elapsed, "us")

    output_buffer.invalidate()
    test_out = output_buffer[:16]
    test_index = np.unpackbits(test_out[12:14],bitorder='little').argmax()
    print("test result:", test_index,"gold result", target_0)
    if test_index != target_0:
        err_cnt = err_cnt + 1

print("accuray: " , 1 - err_cnt/len(image))
print(" ----------------- python schedule finish")

print(" ----------------- c schedule begin")
err_cnt = 0
for i in range(len(image)):
    image_0 = image[i]
    target_0 = target[i]
    print(i)

    start = time.time()
    firefly_in, input_packed = preprocess(model_config_list, image_0, 16)
    input_buffer[:input_packed.size]=input_packed
    input_buffer.flush()
    end = time.time()

    elapsed = round((end - start) * 1000000)
    # print("preprocess:", elapsed, "us")

    input_buffer[:input_packed.size]=input_packed
    input_buffer.flush()
    start = time.time()
    sche(ct_ptr, sche_len)
    end = time.time()

    elapsed = round((end - start) * 1000000)
    # print("clib inference call:", elapsed, "us")

    output_buffer.invalidate()
    test_out = output_buffer[:16]
    test_index = np.unpackbits(test_out[12:14],bitorder='little').argmax()
    print("test result:", test_index,"gold result", target_0)
    if test_index != target_0:
        err_cnt = err_cnt + 1

print("accuray: " , 1 - err_cnt/len(image))
print(" ----------------- c schedule finish")
