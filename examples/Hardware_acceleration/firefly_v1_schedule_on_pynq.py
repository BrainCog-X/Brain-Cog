import numpy as np
import tqdm
from standalone_utils import *
import math
import time
import ctypes as ct


class FireFlyV1ConvSchedule:
    def __init__(
            self,
            ctrl_io,
            allocate_method,
            input_buffer_addr,
            output_buffer_addr,
            weight_data,
            bias_data,

            parallel_channel=16,
            kernel_size=3,
            input_channels=64,
            output_channels=128,
            width=32,
            height=32,
            enable_pooling=False,
            direct_adapt=False,
            winner_takes_all=False,
            final_conv=False,
            time_step=8,
            threshold=64,
            max_cnt=2048
    ):
        self.ctrl_io = ctrl_io

        self.input_buffer_addr = np.uint32(input_buffer_addr)
        self.output_buffer_addr = np.uint32(output_buffer_addr)

        self.weight_buffer = allocate_method(shape=weight_data.size, dtype=np.int8)
        self.bias_buffer = allocate_method(shape=bias_data.size, dtype=np.int16)
        self.weight_buffer_addr = np.uint32(self.weight_buffer.device_address)
        self.bias_buffer_addr = np.uint32(self.bias_buffer.device_address)

        self.weight_buffer[:] = np.ascontiguousarray(weight_data.flatten())
        self.bias_buffer[:] = np.ascontiguousarray(bias_data.flatten())
        self.weight_buffer.flush()
        self.bias_buffer.flush()
        self.max_cnt = max_cnt

        self.parallel_channel = np.uint32(parallel_channel)
        self.kernel_size = np.uint32(kernel_size)
        self.input_channels = np.uint32(input_channels)
        self.output_channels = np.uint32(output_channels)
        self.width = np.uint32(width)
        self.height = np.uint32(height)
        self.enable_pooling = np.uint32(enable_pooling)
        self.direct_adapt = np.uint32(direct_adapt)
        self.winner_takes_all = np.uint32(winner_takes_all)
        self.time_step = np.uint32(time_step)
        self.threshold = np.int32(threshold)

        self.out_width = np.uint32(width >> enable_pooling)
        self.out_height = np.uint32(height >> enable_pooling)

        self.numOfIFMs = np.uint32(input_channels / parallel_channel - 1)
        self.numOfOFMs = np.uint32(output_channels / parallel_channel - 1)
        self.numOfTimeSteps = np.uint32(time_step - 1)
        self.numOfTimeStepIFMs = np.uint32((input_channels / parallel_channel) * time_step - 1)
        self.numOfTimeStepOFMs = np.uint32((output_channels / parallel_channel) * time_step - 1)
        self.weightsLength = np.uint32(input_channels - 1)

        if direct_adapt:
            factor = kernel_size * kernel_size
            padded_length = math.ceil(input_channels / parallel_channel / factor)
            self.numOfIFMs = np.uint32(padded_length - 1)
            self.numOfTimeStepIFMs = np.uint32(padded_length * time_step - 1)
            self.weightsLength = np.uint32(padded_length * parallel_channel - 1)
            self.out_width = np.uint32(1)
            self.out_height = np.uint32(1)

        self.mm2s_fix_len = np.uint32(self.width * self.height * self.time_step * self.input_channels / 8)
        self.s2mm_fix_len = np.uint32(self.out_width * self.out_height * self.parallel_channel / 8)
        self.bias_len = np.uint32(self.output_channels * 2)
        self.weight_len = np.uint32(self.output_channels * self.input_channels * self.kernel_size * self.kernel_size)

        self.stride_of_channel = np.uint32(self.out_width * self.out_height * self.parallel_channel / 8)
        self.stride_of_time_step = np.uint32(self.out_width * self.out_height * self.output_channels / 8)

        if direct_adapt:
            self.mm2s_fix_len = np.uint32(self.time_step * self.input_channels / 8)
            self.weight_len = np.uint32(self.output_channels * self.input_channels)
            self.stride_of_channel = np.uint32(2 * self.parallel_channel / 8)
            self.stride_of_time_step = np.uint32(2 * self.output_channels / 8)

        if final_conv:
            self.flatten_channel = np.uint32(self.out_width * self.out_height * self.output_channels)
            factor = kernel_size * kernel_size * 8 * 4
            round_channel = int(math.ceil(self.flatten_channel / factor) * factor)
            self.stride_of_time_step = np.uint32(round_channel / 8)

        self.configReg_0x00 = np.uint32(((self.time_step - 1) << 16) + (self.numOfOFMs << 8) + self.numOfIFMs).tobytes()
        self.configReg_0x04 = np.uint32((self.numOfTimeStepOFMs << 12) + self.numOfTimeStepIFMs).tobytes()
        self.configReg_0x08 = np.uint32((self.threshold << 14) + self.weightsLength).tobytes()
        self.configReg_0x0c = np.uint32((self.winner_takes_all << 30) + (self.direct_adapt << 29) + (
                self.enable_pooling << 28) + ((self.height - 1) << 16) + self.width - 1).tobytes()
        self.configReg_0x20 = np.uint32(self.stride_of_time_step).tobytes()
        self.configReg_0x24 = np.uint32(self.stride_of_channel).tobytes()
        self.configReg_0x28 = np.uint32(self.input_buffer_addr).tobytes()
        self.configReg_0x2c = np.uint32(self.output_buffer_addr).tobytes()
        self.configReg_0x30 = np.uint32(self.mm2s_fix_len).tobytes()
        self.configReg_0x34 = np.uint32(self.s2mm_fix_len).tobytes()
        self.configReg_0x38 = np.uint32(self.weight_buffer_addr).tobytes()
        self.configReg_0x3c = np.uint32(self.weight_len).tobytes()
        self.configReg_0x40 = np.uint32(self.bias_buffer_addr).tobytes()
        self.configReg_0x44 = np.uint32(self.bias_len).tobytes()

        self.paramCmd = np.uint32(0x00010000).tobytes()
        self.inOutCmd = np.uint32(0x00000101).tobytes()

    def gen_cmd(self):
        cmd_list = []
        cmd_list.append(np.frombuffer(self.configReg_0x00, dtype=np.uint32))
        cmd_list.append(np.frombuffer(self.configReg_0x04, dtype=np.uint32))
        cmd_list.append(np.frombuffer(self.configReg_0x08, dtype=np.uint32))
        cmd_list.append(np.frombuffer(self.configReg_0x0c, dtype=np.uint32))
        cmd_list.append(np.frombuffer(self.configReg_0x20, dtype=np.uint32))
        cmd_list.append(np.frombuffer(self.configReg_0x24, dtype=np.uint32))
        cmd_list.append(np.frombuffer(self.configReg_0x28, dtype=np.uint32))
        cmd_list.append(np.frombuffer(self.configReg_0x2c, dtype=np.uint32))
        cmd_list.append(np.frombuffer(self.configReg_0x30, dtype=np.uint32))
        cmd_list.append(np.frombuffer(self.configReg_0x34, dtype=np.uint32))
        cmd_list.append(np.frombuffer(self.configReg_0x38, dtype=np.uint32))
        cmd_list.append(np.frombuffer(self.configReg_0x3c, dtype=np.uint32))
        cmd_list.append(np.frombuffer(self.configReg_0x40, dtype=np.uint32))
        cmd_list.append(np.frombuffer(self.configReg_0x44, dtype=np.uint32))
        cmd_list.append(np.frombuffer(self.paramCmd, dtype=np.uint32))
        cmd_list.append(np.frombuffer(self.inOutCmd, dtype=np.uint32))
        return np.array(cmd_list).flatten()

    def send_config(self):
        self.ctrl_io.write(0x00, self.configReg_0x00)
        self.ctrl_io.write(0x04, self.configReg_0x04)
        self.ctrl_io.write(0x08, self.configReg_0x08)
        self.ctrl_io.write(0x0c, self.configReg_0x0c)

        self.ctrl_io.write(0x20, self.configReg_0x20)
        self.ctrl_io.write(0x24, self.configReg_0x24)
        self.ctrl_io.write(0x28, self.configReg_0x28)
        self.ctrl_io.write(0x2c, self.configReg_0x2c)
        self.ctrl_io.write(0x30, self.configReg_0x30)
        self.ctrl_io.write(0x34, self.configReg_0x34)
        self.ctrl_io.write(0x38, self.configReg_0x38)
        self.ctrl_io.write(0x3c, self.configReg_0x3c)
        self.ctrl_io.write(0x40, self.configReg_0x40)
        self.ctrl_io.write(0x44, self.configReg_0x44)

    def begin_schedule_non_blocking(self):
        self.ctrl_io.write(0x48, self.paramCmd)
        self.ctrl_io.write(0x48, self.paramCmd)
        self.ctrl_io.write(0x48, self.inOutCmd)

    def begin_schedule_blocking(self):
        self.begin_schedule_non_blocking()
        while self.ctrl_io.read(0x18, length=8) == 0:
            continue
        self.clear_schedule()

    def clear_schedule(self):
        self.ctrl_io.write(0x18, 0)

    def run_all(self):
        self.ctrl_io.write(0x00, self.configReg_0x00)
        self.ctrl_io.write(0x04, self.configReg_0x04)
        self.ctrl_io.write(0x08, self.configReg_0x08)
        self.ctrl_io.write(0x0c, self.configReg_0x0c)

        self.ctrl_io.write(0x20, self.configReg_0x20)
        self.ctrl_io.write(0x24, self.configReg_0x24)
        self.ctrl_io.write(0x28, self.configReg_0x28)
        self.ctrl_io.write(0x2c, self.configReg_0x2c)
        self.ctrl_io.write(0x30, self.configReg_0x30)
        self.ctrl_io.write(0x34, self.configReg_0x34)
        self.ctrl_io.write(0x38, self.configReg_0x38)
        self.ctrl_io.write(0x3c, self.configReg_0x3c)
        self.ctrl_io.write(0x40, self.configReg_0x40)
        self.ctrl_io.write(0x44, self.configReg_0x44)

        self.ctrl_io.write(0x48, self.paramCmd)
        self.ctrl_io.write(0x48, self.paramCmd)
        self.ctrl_io.write(0x48, self.inOutCmd)
        cnt = 0
        while self.ctrl_io.read(0x18, length=8) == 0:
            cnt = cnt + 1
            if cnt > self.max_cnt:
                print("timeout, abort!")
                break
        end = time.time()
        self.ctrl_io.write(0x18, 0)

    def read_status(self):
        status = np.uint32(self.ctrl_io.read(0x50, length=4)).tobytes()
        busy_status = status[0]
        input_status = status[1]
        output_status = status[2]
        param_status = status[3]

        print("busy_status", busy_status)
        print("input_status", input_status)
        print("output_status", output_status)
        print("param_status", param_status)


def create_schedule(model_config_list: list,
                    ctrl_io,
                    allocate_method,
                    buffer_0,
                    buffer_1,
                    image_height,
                    image_width,
                    time_step=4,
                    parallel_channel=16
                    ):
    schedule_list = []
    curr_image_height = image_height
    curr_image_width = image_width
    input_buffer_addr = buffer_0.device_address
    output_buffer_addr = buffer_1.device_address

    for config in model_config_list:
        if config["layer_type"] == "conv+IFNode":
            schedule = FireFlyV1ConvSchedule(
                ctrl_io=ctrl_io,
                allocate_method=allocate_method,

                input_buffer_addr=input_buffer_addr,
                output_buffer_addr=output_buffer_addr,
                weight_data=conv_weight_channel_tiling(parallel_channel, config["weight"]),
                bias_data=config["bias"].flatten(),

                parallel_channel=parallel_channel,
                kernel_size=3,
                input_channels=config["input_channel"],
                output_channels=config["output_channel"],
                width=curr_image_width,
                height=curr_image_height,
                enable_pooling=False,
                time_step=time_step,
                threshold=config["threshold"],
                final_conv="flatten" in config
            )
            schedule_list.append(schedule)
            input_buffer_addr, output_buffer_addr = output_buffer_addr, input_buffer_addr

        elif config["layer_type"] == "conv+IFNode+maxpool":
            schedule = FireFlyV1ConvSchedule(
                ctrl_io=ctrl_io,
                allocate_method=allocate_method,

                input_buffer_addr=input_buffer_addr,
                output_buffer_addr=output_buffer_addr,
                weight_data=conv_weight_channel_tiling(parallel_channel, config["weight"]),
                bias_data=config["bias"].flatten(),

                parallel_channel=parallel_channel,
                kernel_size=3,
                input_channels=config["input_channel"],
                output_channels=config["output_channel"],
                width=curr_image_width,
                height=curr_image_height,
                enable_pooling=True,
                time_step=time_step,
                threshold=config["threshold"],
                final_conv="flatten" in config
            )
            schedule_list.append(schedule)
            input_buffer_addr, output_buffer_addr = output_buffer_addr, input_buffer_addr
            curr_image_height = int(curr_image_height / 2)
            curr_image_width = int(curr_image_width / 2)

        elif config["layer_type"].__contains__("linear"):

            input_channel = config["input_channel"]
            weight = config["weight"]
            if "weight_reshape" in config:
                factor = 9 * 8 * 4
                round_channel = int(math.ceil(input_channel / factor) * factor)
                weight = rearrange(weight, "o (i p h w) -> o (i h w p)", p=parallel_channel,
                                   h=curr_image_height, w=curr_image_width)
                weight = np.pad(weight, ((0, 0), (0, round_channel - input_channel)), mode="constant")
                input_channel = round_channel

            weight = linear_weight_channel_tiling(parallel_channel, weight)

            schedule = FireFlyV1ConvSchedule(
                ctrl_io=ctrl_io,
                allocate_method=allocate_method,

                input_buffer_addr=input_buffer_addr,
                output_buffer_addr=output_buffer_addr,
                weight_data=weight,
                bias_data=config["bias"].flatten(),

                parallel_channel=parallel_channel,
                kernel_size=3,
                input_channels=input_channel,
                output_channels=config["output_channel"],
                width=3,
                height=3,
                enable_pooling=False,
                time_step=time_step,
                threshold=config["threshold"],
                direct_adapt=config["direct_adapt"],
                winner_takes_all=config["winner_take_all"]
            )
            schedule_list.append(schedule)
            input_buffer_addr, output_buffer_addr = output_buffer_addr, input_buffer_addr

    return schedule_list, input_buffer_addr


def schedule_run_all(schedule_list):
    for schedule in schedule_list:
        schedule.run_all()


def gen_cmd_array(schedule_list):
    cmd_array = []
    for schedule in schedule_list:
        cmd_array.append(schedule.gen_cmd().flatten())
    return np.array(cmd_array)


def init_firefly_c_lib(path, schedule_list):
    cmd_arr = gen_cmd_array(schedule_list)
    lib = ct.CDLL(path)
    sche = lib.firefly_v1_schedule
    u32Ptr = ct.POINTER(ct.c_uint32)
    u32PtrPtr = ct.POINTER(u32Ptr)

    ct_arr = np.ctypeslib.as_ctypes(cmd_arr)
    u32PtrArr = u32Ptr * ct_arr._length_
    ct_ptr = ct.cast(u32PtrArr(*(ct.cast(row, u32Ptr) for row in ct_arr)), u32PtrPtr)
    sche_len = ct.c_uint8(cmd_arr.shape[0])

    return sche, ct_ptr, sche_len


def init_firefly_c_lib_with_time(path, schedule_list):
    cmd_arr = gen_cmd_array(schedule_list)
    lib = ct.CDLL(path)
    sche = lib.firefly_v1_schedule_time_it
    u32Ptr = ct.POINTER(ct.c_uint32)
    u32PtrPtr = ct.POINTER(u32Ptr)

    ct_arr = np.ctypeslib.as_ctypes(cmd_arr)
    u32PtrArr = u32Ptr * ct_arr._length_
    ct_ptr = ct.cast(u32PtrArr(*(ct.cast(row, u32Ptr) for row in ct_arr)), u32PtrPtr)
    sche_len = ct.c_uint8(cmd_arr.shape[0])

    return sche, ct_ptr, sche_len


def firefly_v1_simulate(model_config_list, x):
    for config in model_config_list:
        if config["layer_type"] == "input_quant_stub":
            x = np_quantize_prepare(x, config["scale"], config["zero_point"])
        elif config["layer_type"] == "encoder+conv+IFNode":
            x = direct_coding(x, config["weight"], config["bias"], config["time_step"], config["threshold"])
        elif config["layer_type"] == "conv+IFNode":
            x = conv_ifnode_forward(x, config["weight"], config["bias"], config["threshold"])
        elif config["layer_type"] == "conv+IFNode+maxpool":
            x = conv_ifnode_maxpool_forward(x, config["weight"], config["bias"], config["threshold"])
        elif config["layer_type"] == "linear+WTA":
            x = linear_wta_forward(x, config["weight"], config["bias"])
        elif config["layer_type"] == "linear+IFNode":
            x = linear_ifnode_forward(x, config["weight"], config["bias"], config["threshold"])

    return x


def evaluate_simulate(model_config_list, sample):
    correct = 0
    for (image, target) in tqdm.tqdm(zip(sample[0], sample[1]), total=len(sample[0])):
        sim_in = np.expand_dims(image.numpy(), axis=0)
        _, sim_out = firefly_v1_simulate(model_config_list, sim_in)
        correct += sim_out == target.item()
    return correct / len(sample[0])
