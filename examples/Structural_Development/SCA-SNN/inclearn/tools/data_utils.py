import numpy as np


def construct_balanced_subset(x, y):
    xdata, ydata = [], []
    minsize = np.inf
    for cls_ in np.unique(y):
        xdata.append(x[y == cls_])
        ydata.append(y[y == cls_])
        if ydata[-1].shape[0] < minsize:
            minsize = ydata[-1].shape[0]
    for i in range(len(xdata)):
        # if xdata[i].shape[0] < minsize:
            # import pdb
            # pdb.set_trace()
        idx = np.arange(xdata[i].shape[0])
        np.random.shuffle(idx)
        xdata[i] = xdata[i][idx][:minsize]
        ydata[i] = ydata[i][idx][:minsize]
    # !list
    return np.concatenate(xdata, 0), np.concatenate(ydata, 0)