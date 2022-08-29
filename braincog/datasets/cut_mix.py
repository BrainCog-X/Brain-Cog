import math
import numpy as np
import random
from torch.utils.data.dataset import Dataset
from braincog.datasets.rand_aug import SaltAndPepperNoise
import numpy as np
import torch
from torch.nn import functional as F


def event_difference(x1, x2, kernel_size=3):
    padding = kernel_size // 2
    x1 = F.avg_pool2d(x1, kernel_size=kernel_size, stride=1, padding=padding)
    x2 = F.avg_pool2d(x2, kernel_size=kernel_size, stride=1, padding=padding)
    return F.mse_loss(x1, x2)


def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec


def rand_bbox_time(size, rat):
    if len(size) == 4:  # step, channel, height, width
        step = size[0]
    else:
        raise Exception

    cut_t = np.int(step * rat)
    ct = np.random.randint(step)
    bbt1 = np.clip(ct - cut_t // 2, 0, step)
    bbt2 = np.clip(ct + cut_t // 2, 0, step)

    return bbt1, bbt2


def rand_bbox(size, rat):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    else:
        raise Exception

    cut_rat = np.sqrt(rat)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def calc_lam(x1, x2, bbt1, bbt2, bbx1, bbx2, bby1, bby2):
    tot_x1 = x1.sum()
    tot_x2 = x2.sum()
    tot_bb1 = x1[bbt1:bbt2, :, bbx1:bbx2, bby1:bby2].sum()
    tot_bb2 = x2[bbt1:bbt2, :, bbx1:bbx2, bby1:bby2].sum()
    x1_rat = tot_bb1 / tot_x1
    x2_rat = tot_bb2 / tot_x2
    lam = 1. - (x2_rat / (1. - x1_rat + x2_rat))
    return lam


def rand_bbox_st(size, rat):
    temporal_rat = np.random.uniform(rat, 1.)
    wh_rat = rat / temporal_rat
    bbt1, bbt2 = rand_bbox_time(size, temporal_rat)
    bbx1, bby1, bbx2, bby2 = rand_bbox(size, wh_rat)
    return bbt1, bbt2, bbx1, bby1, bbx2, bby2


def spatio_mask(size, rat):
    t = size[0]
    x = torch.rand(2, 2)
    y = torch.rand(2, 2)
    f = torch.zeros(*size[-2:], dtype=torch.complex64)
    # f[0:2, 0:2] = x + y * 0.j
    f[[[0, -1], [-1, -1]], [[0, -1], [0, -1]]] = x + y * 1.j
    mask = torch.fft.ifftn(f).real
    idx = int(np.prod(size[-2:]) * rat)
    val = mask.flatten().sort()[0][idx]

    return (mask < val).unsqueeze(0).unsqueeze(0).repeat(t, 2, 1, 1)


def temporal_mask(size, rat):
    bbt1, bbt2 = rand_bbox_time(size, rat)
    mask = torch.zeros(*size, dtype=torch.bool)
    mask[bbt1:bbt2] = True
    return mask


def st_mask(size, rat):
    t = size[0]
    temporal_rat = np.random.uniform(rat, 1.)
    wh_rat = rat / temporal_rat
    bbt1, bbt2 = rand_bbox_time(size, temporal_rat)
    mask = spatio_mask(size, wh_rat)
    mask[0:bbt1] = False
    mask[bbt2:t] = False
    return mask


def GMM_mask_clip(size, rat):
    t = size[0]
    temporal_rat = np.random.uniform(rat, 1.)
    wh_rat = rat / temporal_rat
    bbt1, bbt2 = rand_bbox_time(size, temporal_rat)
    mask = GMM_mask(size, wh_rat)
    mask[0:bbt1] = False
    mask[bbt2:t] = False
    return mask


def GMM_mask(size, rat, n=None):
    if n is None:
        n = np.random.randint(2, 5)
    pi = torch.tensor(np.random.rand(n))
    # pi = torch.ones(n) / n

    mask = torch.zeros((size[0], size[2], size[3]))
    t = torch.tensor(list(range(size[0])))
    x = torch.tensor(list(range(size[2])))
    y = torch.tensor(list(range(size[3])))
    t, x, y = torch.meshgrid(t, x, y, indexing='ij')

    for p in pi:
        mt = np.random.randint(0, size[0])
        mx = np.random.randint(0, size[2])
        my = np.random.randint(0, size[3])
        # print(mt, mx, my)
        st = max(np.random.rand(), 0.1) * size[0] * 0.5
        sx = max(np.random.rand(), 0.1) * size[2] * .5
        sy = max(np.random.rand(), 0.1) * size[3] * .5
        # st, sx, sy = size[0], 0000.5 * size[2], 0000.5 * size[3]
        # print(st, sx, sy)
        tt = t - mt
        xx = x - mx
        yy = y - my
        tmp = -((tt ** 2) / (st ** 2) + (xx ** 2) / (sx ** 2) + (yy ** 2) / (sy ** 2)) / 2
        mask += p * tmp.exp()

    idx = int(np.prod(mask.shape) * rat)
    val = mask.flatten().sort()[0][idx - 1]
    return (mask > val).unsqueeze(1).repeat(1, 2, 1, 1)
    # return mask.unsqueeze(1).repeat(1, 2, 1, 1)

# FOR EVENT VIS
# def spatio_mask(size, rat):
#     t = size[0]
#     x = torch.rand(2, 2)
#     y = torch.rand(2, 2)
#     f = torch.zeros(*size[-2:], dtype=torch.complex64)
#     # f[0:2, 0:2] = x + y * 0.j
#     f[[[0, -1], [-1, -1]], [[0, -1], [0, -1]]] = x + y * 1.j
#
#     f = f.unsqueeze(0).repeat(t, 1, 1)
#     f[1:-2, :, :] = 0
#
#     mask = torch.fft.ifftn(f).real
#     # print(mask.shape)
#     idx = int(np.prod(mask.shape) * 0.6)
#     # print(idx)
#     val = mask.flatten().sort()[0][idx]
#     print(mask.unsqueeze(1).repeat(1, 2, 1, 1).shape)
#     return (mask < val).unsqueeze(1).repeat(1, 2, 1, 1)
#
# def st_mask(size, rat):
#     # t = size[0]
#     # temporal_rat = np.random.uniform(rat, 1.)
#     # wh_rat = rat / temporal_rat
#     wh_rat = rat
#     # bbt1, bbt2 = rand_bbox_time(size, temporal_rat)
#     mask = spatio_mask(size, wh_rat)
#     # mask[0:bbt1] = False
#     # mask[bbt2:t] = False
#     return mask


def calc_masked_lam(x1, x2, mask):
    tot_x1 = x1.sum()
    tot_x2 = x2.sum()
    tot_mask1 = x1[mask].sum()
    tot_mask2 = x2[mask].sum()
    x1_rat = tot_mask1 / tot_x1
    x2_rat = tot_mask2 / tot_x2
    lam = 1. - (x2_rat / (1. - x1_rat + x2_rat))
    # print(tot_x1, tot_x2, tot_mask1, tot_mask2)
    return lam


def calc_masked_lam_with_difference(x1, x2, mix, kernel_size=3):
    s1 = event_difference(x1, mix, kernel_size=kernel_size)
    s2 = event_difference(x2, mix, kernel_size=kernel_size)
    return (s2 * s2) / (s1 * s1 + s2 * s2)


class MixUp(Dataset):
    def __init__(self, dataset, num_class, num_mix=1, beta=1., prob=1.0, indices=None, noise=0.0, vis=False, **kwargs):
        self.dataset = dataset
        self.num_class = num_class
        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob
        self.indices = indices
        self.noise = noise
        self.vis = vis

    def __getitem__(self, index):
        img, lb = self.dataset[index]
        lb_onehot = onehot(self.num_class, lb)

        if self.vis:
            origin = img.clone()

        for _ in range(self.num_mix):
            r = np.random.rand(1)
            if self.beta <= 0 or r > self.prob:
                continue

            # generate mixed sample
            lam = np.random.beta(self.beta, self.beta)

            if self.indices is None:
                rand_index = random.choice(range(len(self)))
            else:
                rand_index = random.choice(self.indices)

            img2, lb2 = self.dataset[rand_index]
            lb2_onehot = onehot(self.num_class, lb2)

            img = img * lam + img2 * (1. - lam)
            lb_onehot = lb_onehot * lam + lb2_onehot * (1. - lam)

            if self.noise != 0.:
                img = SaltAndPepperNoise(img, self.noise)

        if self.vis:
            return origin, img, img2
        else:
            return img, lb_onehot

    def __len__(self):
        return len(self.dataset)


class CutMix(Dataset):
    #   81.45161290322581 (epoch 584) /data/floyed/BrainCog/train/20220413-050658-resnet34-dvsc10-10-cut_mix before lam
    def __init__(self, dataset, num_class, num_mix=1, beta=1., prob=1.0, indices=None, noise=0.0, vis=False, **kwargs):
        self.dataset = dataset
        self.num_class = num_class
        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob
        self.indices = indices
        self.noise = noise
        self.vis = vis

    def __getitem__(self, index):
        img, lb = self.dataset[index]
        lb_onehot = onehot(self.num_class, lb)

        if self.vis:
            origin = img.clone()

        for _ in range(self.num_mix):
            r = np.random.rand(1)
            if self.beta <= 0 or r > self.prob:
                continue

            # generate mixed sample
            lam = np.random.beta(self.beta, self.beta)

            if self.indices is None:
                rand_index = random.choice(range(len(self)))
            else:
                rand_index = random.choice(self.indices)

            img2, lb2 = self.dataset[rand_index]
            lb2_onehot = onehot(self.num_class, lb2)
            # shape: step, channel, height, width
            # alpha = np.random.rand()

            # if alpha < 0.333:
            bbx1, bby1, bbx2, bby2 = rand_bbox(img.shape, 1. - lam)
            # bbx1, bby1, bbx2, bby2 = 32, 0, 48, 16

            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.shape[-1] * img.shape[-2]))  # area
            # lam = calc_lam(img, img2, 0, shape[0], bbx1, bbx2, bby1, bby2)  # count

            #  distance
            # mask = torch.zeros_like(img, dtype=torch.bool)
            # mask[:, :, bbx1:bbx2, bby1:bby2] = True
            # mix = img.clone()
            # mix[mask] = img2[mask]
            # lam = calc_masked_lam_with_difference(img, img2, mix, kernel_size=3)
            if self.vis:
                img[:, :, bbx1:bbx2, bby1:bby2] = -img2[:, :, bbx1:bbx2, bby1:bby2]
                img2 = -img2
            else:
                img[:, :, bbx1:bbx2, bby1:bby2] = img2[:, :, bbx1:bbx2, bby1:bby2]

            # elif alpha > 0.667:
            #     bbt1, bbt2 = rand_bbox_time(img.shape, 1. - lam)
            #     lam = calc_lam(img, img2, bbt1, bbt2, 0, shape[2], 0, shape[3])
            #     img[:, bbt1:bbt2, :, :] = img2[:, bbt1:bbt2, :, :]
            #     # lam = 1 - (bbt2 - bbt1) / (img.shape[-4])
            # else:
            #     bbt1, bbt2, bbx1, bby1, bbx2, bby2 = rand_bbox_st(img.shape, 1. - lam)
            #     lam = calc_lam(img, img2, bbt1, bbt2, bbx1, bbx2, bby1, bby2)
            #     img[:, bbt1:bbt2, bbx1:bbx2, bby1:bby2] = img2[:, bbt1:bbt2, bbx1:bbx2, bby1:bby2]
            #     # lam = 1 - ((bbt2 - bbt1) * (bbx2 - bbx1) * (bby2 - bby1) / (img.shape[-1] * img.shape[-2] * img.shape[-4]))

            if self.noise != 0.:
                img = SaltAndPepperNoise(img, self.noise)
            lb_onehot = lb_onehot * lam + lb2_onehot * (1. - lam)

        if self.vis:
            mask = torch.zeros_like(img)
            mask[:, :, bbx1:bbx2, bby1:bby2] = 1.
            return origin, img, img2, mask
        else:
            return img, lb_onehot

    def __len__(self):
        return len(self.dataset)


class EventMix(Dataset):
    #  82.15725806451613 (epoch 554) /data/floyed/BrainCog/train/20220413-014843-resnet34-dvsc10-10-masked
    def __init__(self,
                 dataset,
                 num_class,
                 num_mix=1,
                 beta=1.,
                 prob=1.0,
                 indices=None,
                 noise=0.1,
                 vis=False,
                 gaussian_n=None,
                 **kwargs):
        self.dataset = dataset
        self.num_class = num_class
        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob
        self.indices = indices
        self.noise = noise
        self.vis = vis
        self.gaussian_n = gaussian_n
        print(self.prob, self.gaussian_n, self.beta)

    def __getitem__(self, index):
        img, lb = self.dataset[index]
        lb_onehot = onehot(self.num_class, lb)

        shape = img.shape
        if self.vis:
            origin = img.clone()

        for _ in range(self.num_mix):
            r = np.random.rand(1)
            if self.beta <= 0 or r > self.prob:
                continue

            # generate mixed sample
            lam = np.random.beta(self.beta, self.beta)  # lam -> remain ratio

            if self.indices is None:
                rand_index = random.choice(range(len(self)))
            else:
                rand_index = random.choice(self.indices)

            img2, lb2 = self.dataset[rand_index]
            lb2_onehot = onehot(self.num_class, lb2)
            # shape: step, channel, height, width
            # alpha = np.random.rand()
            # if alpha < 0.333:
            # mask = spatio_mask(shape, 1. - lam)
            # elif alpha > 0.667:
            # mask = temporal_mask(shape, 1. - lam)
            # else:
            # mask = st_mask(shape, 1. - lam)
            mask = GMM_mask(shape, 1. - lam, self.gaussian_n)
            # mask = GMM_mask_clip(shape, 1. - lam)
            # mask = torch.logical_not(mask)

            # lam = 1 - (mask.sum() / np.prod(img.shape))  #  area
            lam = calc_masked_lam(img, img2, mask)  # count
            img[mask] = img2[mask]  # count && mask required

            # distance
            # mix = torch.clone(img)
            # if self.vis:
            #     mix[mask] = -img2[mask]
            #     img2 = -img2
            # else:
            #     mix[mask] = img2[mask]
            # lam = calc_masked_lam_with_difference(img, img2, mix, kernel_size=3)
            # img = mix

            if self.noise != 0.:
                img = SaltAndPepperNoise(img, self.noise)

            lb_onehot = lb_onehot * lam + lb2_onehot * (1. - lam)

        if self.vis:
            return origin, img, img2, mask
        else:
            return img, lb_onehot

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d import proj3d


    def get_proj(self):
        """
         Create the projection matrix from the current viewing position.

         elev stores the elevation angle in the z plane
         azim stores the azimuth angle in the (x, y) plane

         dist is the distance of the eye viewing point from the object point.
        """
        # chosen for similarity with the initial view before gh-8896

        relev, razim = np.pi * self.elev / 180, np.pi * self.azim / 180

        # EDITED TO HAVE SCALED AXIS
        xmin, xmax = np.divide(self.get_xlim3d(), self.pbaspect[0])
        ymin, ymax = np.divide(self.get_ylim3d(), self.pbaspect[1])
        zmin, zmax = np.divide(self.get_zlim3d(), self.pbaspect[2])

        # transform to uniform world coordinates 0-1, 0-1, 0-1
        worldM = proj3d.world_transformation(xmin, xmax,
                                             ymin, ymax,
                                             zmin, zmax)

        # look into the middle of the new coordinates
        R = self.pbaspect / 2

        xp = R[0] + np.cos(razim) * np.cos(relev) * self.dist
        yp = R[1] + np.sin(razim) * np.cos(relev) * self.dist
        zp = R[2] + np.sin(relev) * self.dist
        E = np.array((xp, yp, zp))

        self.eye = E
        self.vvec = R - E
        self.vvec = self.vvec / np.linalg.norm(self.vvec)

        if abs(relev) > np.pi / 2:
            # upside down
            V = np.array((0, 0, -1))
        else:
            V = np.array((0, 0, 1))
        zfront, zback = -self.dist, self.dist

        viewM = proj3d.view_transformation(E, R, V)
        projM = self._projection(zfront, zback)
        M0 = np.dot(viewM, worldM)
        M = np.dot(projM, M0)
        return M


    Axes3D.get_proj = get_proj

    size = (100, 2, 48, 48)
    mask = GMM_mask(size, 0.3)
    print(mask.shape)
    # for i in range(100):
    #     plt.figure()
    #     plt.imshow(mask[i, 0])
    # plt.show()

    pos_idx1 = []
    neg_idx1 = []
    for t in range(100):
        for r in range(48):
            for c in range(48):
                if mask[t, 0, r, c] > 0:
                    pos_idx1.append((t, r, c))
                if mask[t, 1, r, c] > 0:
                    neg_idx1.append((t, r, c))
    pos_t1, pos_x1, pos_y1 = np.split(np.array(pos_idx1), 3, axis=1)
    neg_t1, neg_x1, neg_y1 = np.split(np.array(neg_idx1), 3, axis=1)

    fig = plt.figure(figsize=plt.figaspect(0.5) * 1.5)
    ax = Axes3D(fig)
    ax.pbaspect = np.array([1, 1, 1])  # np.array([2.0, 1.0, 0.5])
    ax.view_init(elev=10, azim=-75)
    # ax.axis('off')
    ax.scatter(pos_t1[:, 0], pos_y1[:, 0], 48 - pos_x1[:, 0], color='red', alpha=0.1, s=2.)
    ax.scatter(neg_t1[:, 0], neg_y1[:, 0], 48 - neg_x1[:, 0], color='blue', alpha=0.1, s=2.)
    plt.show()
