import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def unit(x):
    if len(x.shape)>0:
        maxx=np.max(x)
        minx=np.min(x)
        marge=maxx-minx
        if marge!=0:
            xx=(x-minx)/marge
            xx=np.clip(xx, 0,1)
        else:
            xx=0.5*np.ones_like(x)
        return xx
    else:
        return x

def unit_tensor(x):
    if x.size()[0]>0:
        maxx=torch.max(x)
        minx=torch.min(x)
        marge=maxx-minx
        if marge!=0:
            xx=(x-minx)/marge
        else:
            xx=0.5*torch.ones_like(x)
        return xx
    else:
        return x