
import torch
from tools.LSM_helper import update_matrix_to_list, draw_spikes

def regul(matrix,outputsize=3):
    n_offs=matrix.size()[0]
    maxvalue=torch.zeros([n_offs])
    minvalue=torch.zeros(n_offs)
    ze=torch.zeros_like(matrix[0])
    for i in range(n_offs):
        maxvalue[i]=matrix[i].max()
        mask=matrix[i]>0
        minvalue[i]=matrix[i][mask].min()

    for i in range(n_offs):
        if maxvalue[i]!=0:
            # matrix[i,:,j]=torch.where(matrix[i,:,j]>0,torch.div(matrix[i,:,j]-minvalue[i][j],maxvalue[i][j]-minvalue[i][j]),ze)
            matrix[i]=matrix[i]*matrix[i].nonzero().size()[0]/matrix[i].sum()

    return matrix


def bcm(model,raw_reward,input,bcm_reward_scale=0.0005,):
    # w_scale=0.5
    num_l=model.num_of_liquid_layer_neurons
    raw_reward=torch.from_numpy(raw_reward).to(model.device).float()
    reward = torch.ones(size=[model.n_offsprings, model.n_output]).to(model.device) * (-raw_reward).unsqueeze(1)
    reward[[i for i in range(model.n_offsprings)],model.out]=raw_reward
    reward=torch.unsqueeze(reward, 1)
    # prod=torch.zeros(self.n_offspring*self.n_pseudo_env,num_l,num_l).to(self.device)
    # sumspikes = self.sumspikes[0]
    # thre = self.thre[0]
    # for i in range(num_l):
    #     prod[:,:,i]=torch.unsqueeze(sumspikes[:,i],1)*sumspikes*(sumspikes-thre) #off*liquid*liquid
    # delta_w=prod-0.95*self.liquid_weight_matrix

    # delta_w=delta_w*reward*w_scale
    # self.liquid_weight_matrix=self.liquid_weight_matrix+delta_w
    # self.liquid_weight_matrix=(self.liquid_weight_matrix-self.liquid_weight_matrix.min())/(self.liquid_weight_matrix.max()-self.liquid_weight_matrix.min())

######################### train readout layer
    num_o=model.n_output
    prod=torch.zeros(model.n_offsprings,num_l,num_o).to(model.device)
    liquid_spikes = model.sumspikes[0]
    output_spikes = model.sumspikes[1]

    thre = model.thre[1]
    for i in range(model.n_offsprings):
        # delta=y(y-theta)x-sigma*w,prod=y(y-theta)x
        # pre:x,output,post:y,liquid
        x=liquid_spikes[i]
        y=output_spikes[i]
        if(((y-thre[i])==0).all()):
            prod[i]=torch.matmul(x.unsqueeze(1),(y.unsqueeze(0)))
        else:
            prod[i]=torch.matmul(x.unsqueeze(1),(y*(y-thre[i]).unsqueeze(0)))
        if prod[i].abs().max()!=0:
            prod[i]=torch.div(prod[i], prod[i].abs().max())
    prod = prod*model.readout_mask
    delta_w=prod.abs()*reward
    # draw_spikes(model, id=2, inputsize=4, i_s=input,l_s=model.sumspikes[0], r_s=model.sumspikes[1])
    model.liquid_to_output_weight_matrix+=delta_w

    # model.liquid_to_output_weight_matrix=torch.clamp(model.liquid_to_output_weight_matrix,min=0.001)

    # model.liquid_to_output_weight_matrix=model.liquid_to_output_weight_matrix.masked_fill(model.readout_mask, 0)
    # model.liquid_to_output_list=update_matrix_to_list(model.liquid_to_output_weight_matrix.permute(0, 2, 1),id=model.num_of_liquid_layer_neurons+model.n_input)
    # draw_spikes(model, id=8, inputsize=4, i_s=input,l_s=model.sumspikes[0], r_s=model.sumspikes[1])
    # self.liquid_to_output_weight_matrix+=(~self.readout_mask).int()*0.001

    model.liquid_to_output_weight_matrix=regul(model.liquid_to_output_weight_matrix)
    # draw_spikes(model, id=8, inputsize=4, i_s=input,l_s=model.sumspikes[0], r_s=model.sumspikes[1])
    model.liquid_to_output_weight_matrix *=model.w_output_scale
    model.liquid_to_output_list=update_matrix_to_list(model.liquid_to_output_weight_matrix.permute(0, 2, 1),id=model.num_of_liquid_layer_neurons+model.n_input)

    # draw_spikes(model, id=2, inputsize=4, i_s=input,l_s=model.sumspikes[0], r_s=model.sumspikes[1])

def stdp(model, input,stdp_lr=0.005):
    num_l=model.num_of_liquid_layer_neurons
    prod=torch.zeros(model.n_offsprings,num_l,num_l).to(model.device)
    liquid_spikes = model.sumspikes[0]
    mask=(model.liquid_weight_matrix!=0).to(int)
    for i in range(model.n_offsprings):
        # delta=y(y-theta)x-sigma*w,prod=y(y-theta)x
        # pre:x,output,post:y,liquid
        x=liquid_spikes[i]

        prod[i]=torch.matmul(x.unsqueeze(1),x.unsqueeze(0))*mask[i]
        if prod[i].abs().max()!=0:
            prod[i]=torch.div(prod[i], prod[i].abs().max())
    prod = prod.masked_fill(model.liquid_mask, 0)
    delta_w=prod
    model.liquid_weight_matrix+=delta_w
    model.liquid_weight_matrix *=model.w_liquid_scale
    # draw_spikes(model, id=2, i_s=input,inputsize=4,l_s=model.sumspikes[0], r_s=model.sumspikes[1])
