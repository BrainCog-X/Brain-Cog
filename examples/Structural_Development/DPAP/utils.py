import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_log(print_string, log):
    #print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()

def unit(x):
    if x.size()[0]>0:
        xnp=x.cpu().numpy()
        maxx=np.percentile(xnp, 75)
        minx=torch.min(x)
        marge=maxx-minx
        if marge!=0:
            xx=(x-minx)/marge
            xx=torch.clip(xx, 0,1)
        else:
            xx=0.5*torch.ones_like(x)
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

def init(batch,convlayer,fclayer,size,fcsize):
    neuron_th={}
    convtra = {}
    bcm={}
    epoch_trace = {}
    for i in range(1,len(convlayer)):
        index=convlayer[i]
        neuron_th[index]=torch.zeros((batch,size[i]),device=device)
        convtra[index] = torch.zeros(size[i],device=device)
        bcm[index]=torch.zeros(size[i],size[i-1],device=device)
        epoch_trace[index] = torch.zeros((size[i]),device=device)
    for i in range(1,len(fclayer)):
        index=fclayer[i]
        neuron_th[index]=torch.zeros((batch,fcsize[i]),device=device)
        convtra[index]=torch.zeros(fcsize[i],device=device)
        bcm[index]=torch.zeros(fcsize[i],fcsize[i-1],device=device)
        epoch_trace[index] = torch.zeros(fcsize[i],device=device)
    return neuron_th,convtra,bcm,epoch_trace
