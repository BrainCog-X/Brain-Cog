import random
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional
from torchvision.transforms import InterpolationMode


def ShearX(x, v):  # [-0.3, 0.3]
    assert 0 <= v <= 30
    v = np.random.uniform(0, v)
    if random.random() > 0.5:
        v = -v
    return functional.affine(x, angle=0, translate=[0, 0], scale=1., shear=[v, 0])


def ShearY(x, v):  # [-0.3, 0.3]
    assert 0 <= v <= 30
    v = np.random.uniform(0, v)
    if random.random() > 0.5:
        v = -v
    return functional.affine(x, angle=0, translate=[0, 0], scale=1., shear=[0, v])


def TranslateX(x, v):
    assert 0 <= v <= 0.45
    v = np.random.uniform(0, v)
    w, h = x.shape[-2::]
    v = round(w * v)
    if random.random() > 0.5:
        v = -v
    return functional.affine(x, angle=0, translate=[0, v], scale=1., shear=[0, 0])


def TranslateY(x, v):
    assert 0 <= v <= 0.45
    v = np.random.uniform(0, v)
    w, h = x.shape[-2::]
    v = round(w * v)
    if random.random() > 0.5:
        v = -v
    return functional.affine(x, angle=0, translate=[v, 0], scale=1., shear=[0, 0])


def Rotate(x, v):  # [-30, 30]
    assert 0 <= v <= 30
    v = np.random.uniform(0, v)
    if random.random() > 0.5:
        v = -v
    return functional.affine(x, angle=v, translate=[0, 0], scale=1., shear=[0, 0])


def CutoutAbs(x, v):  # [0, 60] => percentage: [0, 0.2]
    assert 0 <= v <= 0.5
    w, h = x.shape[-2::]
    v = round(v * w)

    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = round(max(0, x0 - v / 2.))
    y0 = round(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    x[:, :, y0:y1, x0:x1] = 0.
    return x


def CutoutTemporal(x, v):
    assert 0 <= v <= 0.5
    v = np.random.uniform(0, v)
    step = x.shape[0]
    v = round(v * step)
    t0 = np.random.randint(step)
    t1 = min(step, t0 + v)
    x[t0:t1, :, :, :] = 0.
    return x


def TemporalShift(x, v):
    # TODO: Maybe shift too mach than origin has
    assert 0 <= v <= 0.2
    v = v / 2.
    shape = x.shape
    # p = torch.zeros(2 * (shape[0] - 1), *shape[-3:], device=x.device)
    shift = []
    for i in range(x.shape[0] - 1):
        spike = x[i].clone()
        _max = int(spike.max())
        sft = torch.zeros(shape[-3:], device=x.device)
        for j in range(_max):
            p = torch.rand_like(sft)
            sft[torch.logical_and(p < v, spike > 0.)] += 1.
            spike -= 1
        shift.append(sft)

        spike = x[i + 1].clone()
        _max = int(spike.max())
        sft = torch.zeros(shape[-3:], device=x.device)
        for j in range(_max):
            p = torch.rand_like(sft)
            sft[torch.logical_and(p < v, spike > 0.)] += 1.
        shift.append(sft)

    for i in range(shape[0] - 1):
        sft_next = shift[i * 2]
        sft_pre = shift[i * 2 + 1]
        x[i + 1] = torch.clip(x[i + 1] + sft_next - sft_pre, 0.)
        x[i] = torch.clip(x[i] - sft_next + sft_pre, 0.)

    return x


def SpatioShift(x, v):
    # assert 0 <= v <= 0.1
    w, h = x.shape[-2::]
    shift_x = round(random.uniform(-v, v) * w)
    shift_y = round(random.uniform(-v, v) * h)
    output = []
    step = x.shape[0]
    for t in range(step):
        output.append(functional.affine(x[t],
                                        angle=0,
                                        translate=[
                                            round(shift_x * t / step),
                                            round(shift_y * t / step)],
                                        scale=1.,
                                        shear=[0, 0]))
    return torch.stack(output, dim=0)


def drop(x, v):
    assert 0 <= v <= 0.5
    v = np.random.uniform(0, v)
    _max = int(torch.max(x))
    p = torch.rand((_max, *x.shape), device=x.device)

    for i in range(_max):
        p[i, x > 0] += 1.
        x -= 1.

    p = torch.where(p > 1. + v, 1., 0.)
    return torch.sum(p, dim=0)


def GaussianBlur(x, v):
    assert 0.1 <= v <= 1.
    v = np.random.uniform(0.1, v)
    return functional.gaussian_blur(x, kernel_size=[5, 5], sigma=v)


def SaltAndPepperNoise(x, v):
    assert 0 <= v <= 0.3
    v = np.random.uniform(0, v)
    p = torch.rand_like(x)
    p = torch.where(p > v, 0., 1.)
    return x + p


def Identity(x, v):
    return x

#                           DVSC10 NCAL
augment_list = [  # normal: 79.44  77.57
    # (ShearX, 0, 20),  # 75.71
    # (ShearY, 0, 20),
    # (TranslateX, 0, 0.25),  # 77.52
    # (TranslateY, 0, 0.25),
    # (Rotate, 0, 30),  # 77.02
    (CutoutAbs, 0, 0.5),  # 79.13
    (CutoutTemporal, 0, 0.5),  # 80.65
    # (TemporalShift, 0, 0.2), # 75.30
    # (SpatioShift, 0, 0.1),   # 78.43
    (GaussianBlur, 0, 1.),  # 79.83
    # (drop, 0, 0.5),  # 74.00
    (SaltAndPepperNoise, 0, 0.3),  # 79.64
    # cutmix_normal_aug: 90.02  86.52
]


class RandAugment:
    def __init__(self, n, m):
        self.n = n
        self.m = m      # [0, 30]
        self.augment_list = augment_list

    def __call__(self, x):
        ops = random.choices(self.augment_list, k=self.n)
        for op, minvalue, maxvalue in ops:
            val = (float(self.m) / 30) * float(maxvalue - minvalue) + minvalue
            x = op(x, val)

        return x
