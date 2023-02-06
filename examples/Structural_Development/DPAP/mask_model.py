import abc
from functools import partial
from torch.nn import functional as F
import torchvision
from timm.models import register_model

from braincog.base.node.node import *
from braincog.base.encoder.encoder import *
from braincog.model_zoo.base_module import BaseModule, BaseConvModule, BaseLinearModule
from utils import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

convlayer = [-1,0, 1, 3, 4, 6, 7]
fclayer=[8,9]
imgsize = [32,32, 32, 16,16, 16, 8,8, 8]
size = [3,128, 128, 256, 256, 512,512]
size_pool = [3,128, 128,128, 256, 256,256, 512,512]
fcsize=[512*8*8,512]

class my_cifar_model(BaseModule):
    def __init__(self,
                 num_classes=10,
                 step=8,
                 node_type=LIFNode,
                 encode_type='direct',
                 *args,
                 **kwargs):
        super().__init__(step, encode_type, *args, **kwargs)

        self.num_classes = num_classes

        # self.node = node_type
        # if issubclass(self.node, BaseNode):
        #     self.node = partial(self.node, **kwargs, step=step)


        self.feature = nn.Sequential(
            BaseConvModule(size[0], size[1], kernel_size=(3, 3), padding=(1, 1)),
            BaseConvModule(size[1],size[2], kernel_size=(3, 3), padding=(1, 1)),
            nn.MaxPool2d(2),
            BaseConvModule(size[2], size[3], kernel_size=(3, 3), padding=(1, 1)),
            BaseConvModule(size[3], size[4], kernel_size=(3, 3), padding=(1, 1)),
            nn.MaxPool2d(2),
            BaseConvModule(size[4], size[5], kernel_size=(3, 3), padding=(1, 1)),
            BaseConvModule(size[5], size[6], kernel_size=(3, 3), padding=(1, 1)),
        )
        self.cfla=self._cflatten()
        self.fc_prun = self._create_fc_prun()
        self.fc = self._create_fc()

    def _cflatten(self):
        fc = nn.Sequential(
            nn.Flatten(),
        )
        return fc
        
    def _create_fc_prun(self):
        fc = nn.Sequential(
            BaseLinearModule(fcsize[0], fcsize[1])
        )
        return fc

    def _create_fc(self):
        fc = nn.Sequential(
            BaseLinearModule(fcsize[1], self.num_classes)
        )
        return fc
    
    def forward(self, inputs):
        inputs = self.encoder(inputs)

        self.reset()
        if not self.training:
            self.fire_rate.clear()

        outputs = []
        spikes=[]
                
        for t in range(self.step):
            spikest=[]
            x = inputs[t]
            if x.shape[-1] > 32:
                x = F.interpolate(x, size=[64, 64])
            spikest.append(x.detach())
            for i in range(len(self.feature)):
                spikei=self.feature[i](x)
                x=spikei
                spikest.append(spikei.detach())

            x=self.cfla(x)
            spikest.append(x.detach())
            x=self.fc_prun(x)
            spikest.append(x.detach())
            x = self.fc(x)
            spikes.append(spikest)

            outputs.append(x)

        return sum(outputs) / len(outputs),spikes


class Mask:
    def __init__(self, model,batch,step):
        self.model = model
        self.fullbook={}
        self.mat = {}
        self.feature=model.feature
        self.fc={}
        self.fc[1]=model.fc_prun[0]
        self.fc[2]=model.fc[0]
        self.n_delta={}
        self.ww_delta={}
        self.reduce={}
        self.reduceww={}
        self.batch=batch
        self.step=step

    def init_length(self):
        for i in range(1,len(convlayer)):
            index=convlayer[i]
            self.fullbook[index] =torch.ones((size[i],size[i-1],3,3),device=device)
            self.n_delta[index]=torch.zeros(size[i],device=device)
            self.reduce[index] = 10*torch.ones(size[i],device=device)
        for i in range(1,len(fclayer)):
            index=fclayer[i]
            self.fullbook[index] = torch.ones((fcsize[i],fcsize[i-1]),device=device)
            self.n_delta[index]=torch.zeros(fcsize[i],device=device)
            self.ww_delta[index]=torch.zeros(fcsize[i]*fcsize[i-1],device=device)
            self.reduce[index] = 10*torch.ones(fcsize[i],device=device)
            self.reduceww[index] = 10*torch.ones(fcsize[i]*fcsize[i-1],device=device)
            
            
    def get_filter_codebook(self,ww,dendrite,ii,index,epoch): 
        if ii == 4:
            wconv= dendrite#.cpu().numpy()
            self.n_delta[index]=(unit(wconv)*2-0.65)
            pos=torch.nonzero(self.n_delta[index]>0)
            self.n_delta[index][pos]=self.n_delta[index][pos]+5
            print(wconv.mean(),wconv.max(), wconv.min())
            self.reduce[index]=self.reduce[index]*0.999+self.n_delta[index]*math.exp(-int((epoch-5)/12))
            filter_ind = torch.nonzero(self.reduce[index] <0)
            print(self.reduce[index].mean(),self.reduce[index].max(),self.reduce[index].min(),len(filter_ind))
             
            for x in range(0, len(filter_ind)):
                self.fullbook[index][filter_ind[x]] = 0
      
        if ii == 2:
            length=ww.size()[0]*ww.size()[1]
            book=torch.ones(length,device=device)
            filter_ww = ww.view(-1)#.cpu().numpy()
            self.ww_delta[index]=(unit(filter_ww)*2-1.5)
            pos=torch.nonzero(self.ww_delta[index]>0)
            self.ww_delta[index][pos]=self.ww_delta[index][pos]+2
            self.reduceww[index]= self.reduceww[index]*0.999+self.ww_delta[index]*math.exp(-int((epoch-5)/13))
            filter_indww =torch.nonzero(self.reduceww[index] < 0)
            book[filter_indww]=0
            book=book.reshape((ww.size()[0],-1))
            self.fullbook[index]=self.fullbook[index]*book
            print(self.reduceww[index].mean(),self.reduceww[index].max(),self.reduceww[index].min(),len(filter_indww))
                
            wconv= dendrite#.cpu().numpy()
            self.n_delta[index]=(unit(wconv)*2-1.5)
            pos=torch.nonzero(self.n_delta[index]>0)
            self.n_delta[index][pos]=self.n_delta[index][pos]+2
            self.reduce[index]=self.reduce[index]*0.999+self.n_delta[index]*math.exp(-int((epoch-5)/13))
            filter_ind = torch.nonzero(self.reduce[index] <0)
            print(self.reduce[index].mean(),self.reduce[index].max(),self.reduce[index].min(),len(filter_ind))
             
            for x in range(0, len(filter_ind)):
                self.fullbook[index][filter_ind[x]] = 0

        return self.fullbook[index]

    def convert2tensor(self, x):
        x = torch.FloatTensor(x)
        return x

    def init_mask(self, wwfc,convtra,epoch):
        for i in range(1,len(convlayer)):
            index=convlayer[i]
            ww = wwfc[index]
            dendrite=convtra[index]
            self.mat[index]=self.get_filter_codebook(ww, dendrite,4,index,epoch)
            #self.mat[index] = self.convert2tensor(self.mat[index]).cuda()
        for i in range(1,len(fclayer)):
            index=fclayer[i]
            ww=wwfc[index]
            dendrite=convtra[index]
            self.mat[index]=self.get_filter_codebook(ww,dendrite,2,index,epoch)
            #self.mat[index] = self.convert2tensor(self.mat[index]).cuda()

    def do_mask(self):
        for i in range(1,len(convlayer)):
            index=convlayer[i]
            ww = self.feature[index].conv.weight
            maskww=ww*self.mat[index]
            self.feature[index].conv.weight.data=maskww
        for i in range(1,len(fclayer)):
            ind=fclayer[i]
            ww = self.fc[i].fc.weight
            maskww=ww*self.mat[ind]
            self.fc[i].fc.weight.data=maskww

    def if_zero(self):
        cc=[]
        for i in range(1,len(convlayer)):
            ww=self.feature[convlayer[i]].conv.weight
            b = ww.data.view(-1).cpu().numpy()
            print("number of weight is %d, zero is %.3f" %(len(b),100*(len(b)- np.count_nonzero(b))/len(b)))
            cc.append(100*(len(b)- np.count_nonzero(b))/len(b))
        for i in range(1,len(fcsize)):
            ww=self.fc[i].fc.weight
            b = ww.data.view(-1).cpu().numpy()
            print("number of weight is %d, zero is %.3f" %(len(b),100*(len(b)- np.count_nonzero(b))/len(b)))
            cc.append(100*(len(b)- np.count_nonzero(b))/len(b))
        return cc

class Trace:
    def __init__(self, model,batch,step):
        self.model = model
        self.feature=model.feature
        self.ctrace={}
        self.fctrace={}
        self.csum={}
        self.fcsum={}
        self.delta = 0.5
        self.batch=batch
        self.step=step

    def computing_trace(self,spikes):
        for i in range(len(imgsize)):
            index=i-1
            self.ctrace[index]=torch.zeros((self.batch,size_pool[i],imgsize[i],imgsize[i]),device=device)
        for i in range(len(fclayer)):
            index=fclayer[i]
            self.fctrace[index]=torch.zeros((self.batch,fcsize[i]),device=device)
        for t in range(self.step):      
            for i in range(len(imgsize)):
                index=i-1
                sp=spikes[t][index+1].detach()
                #print(sp.size(),self.ctrace[index].size())
                self.ctrace[index]=self.delta*self.ctrace[index].cuda()+sp.cuda()
            for i in range(len(fclayer)):
                index=fclayer[i]
                sp=spikes[t][index+1].detach()
                self.fctrace[index]=self.delta*self.fctrace[index].cuda()+sp.cuda()
        for i in range(len(imgsize)):
            index=i-1
            self.csum[index]=self.ctrace[index]/(self.step)
            self.csum[index]=torch.sum(torch.sum(self.csum[index],dim=2),dim=2)
        for i in range(len(fclayer)):
            index=fclayer[i]
            self.fcsum[index]=self.fctrace[index]/(self.step)
        return self.csum,self.fcsum
