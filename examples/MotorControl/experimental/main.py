import torch
import numpy as  np
import torch.nn as nn
from model import Motion
import tqdm
import argparse
from torch.nn import functional as F

parser = argparse.ArgumentParser(description='Motor Parameters')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--time-window', type=int, default=8, help="Number of timesteps to do.")
parser.add_argument('--device', type=str, default='0', help="CUDA device")
parser.add_argument('--log-path', type=str, default='./logs/out.txt', help="Log path")

args = parser.parse_args()
print(args)

device = torch.device('cuda:'+args.device)
LABELS = {
    'position_group_0':  (-0.337, -0.020,  -0.077,  -0.031164,  0.999496, -0.005979,  2.850154),
    'position_group_1':  (-0.337,  0.007,  -0.077,  -0.039668,  0.999161, -0.010174,  2.894892),
    'position_group_2':  (-0.337,  0.030,  -0.076,  -0.031164,  0.999496, -0.005979,  2.850154),
    'position_group_3':  (-0.337,  0.052,  -0.076,  -0.031164,  0.999496, -0.005979,  2.850154),
    'position_group_4':  (-0.339,  0.074,  -0.076,   0.016057,  0.999842, -0.007643,  2.804204),    
    'position_group_5':  (-0.339,  0.096,  -0.078,   0.016057,  0.999842, -0.007643,  2.804204),  
    'position_group_6':  (-0.339,  0.123,  -0.079,   0.016057,  0.999842, -0.007643,  2.804204),  
    'position_group_7':  (-0.337,  0.139,  -0.080,   0.076912,  0.997035, -0.0021723, 2.799101),  
    'position_group_8':  (-0.337,  0.163,  -0.0770,  0.076912,  0.997035, -0.0021723, 2.799101),  
    'position_group_9':  (-0.338,  0.188,  -0.075,   0.076912,  0.997035, -0.002172,  2.799101),  
    'position_group_10': (-0.338,  0.212,  -0.075,   0.087103,  0.995757, -0.029681,  2.785759),  
    'position_group_11': (-0.338,  0.235,  -0.070,   0.087103,  0.995757, -0.029681,  2.785759),  
    'position_group_12': (-0.338,  0.259,  -0.073,   0.087103,  0.995757, -0.029681,  2.785759),  
    'position_group_13': (-0.339,  0.273,  -0.065,   0.202020,  0.979225,  0.017483,  2.764647),  
    'position_group_14': (-0.336,  0.290,  -0.066,   0.244628,  0.963147, -0.111827,  2.740450),  
}
position_num = 15
position_dims = 7

TARGETS = []
for i in range(position_num):
    TARGETS.append(np.array(LABELS['position_group_'+str(i)], dtype=np.float32))
TARGETS = np.stack(TARGETS, axis=0)


t_factors = np.array([10.0, 10.0, 100.0, 10.0, 1.0, 100.0, 1.0], dtype=np.float32)
TARGETS_FAC = TARGETS * t_factors[np.newaxis, :]

KEYS = {
    'c1': 0,
    'd2': 1,
    'e1': 2,
    'f1': 3,
    'g1': 4, 
    'a1': 5,
    'b1': 6,
    'c2': 7,
    'd2': 8,
    'e2': 9,
    'f2': 10,
    'g2': 11,
    'a2': 12,
    'b2': 13,
    'c3': 14,
    'd3': 15,
    'e3': 16
}

finger_num = 3
finger_pop_num = 10
key_num = 17
key_pop_num = 5
def creat_key_finger_emb():
    key_value = np.zeros((key_num, key_num*key_pop_num), dtype=np.float32)
    finger_value = np.zeros((finger_num, finger_num*finger_pop_num), dtype=np.float32)
    for i in range(key_num):
        key_value[i, i*key_pop_num: (i+1)*key_pop_num] += 1.0   
    for i in range(finger_num):
        finger_value[i, i*finger_pop_num: (i+1)*finger_pop_num] += 1.0  
    return (key_value, finger_value)

def mse_loss(pred, target):
    mse = F.mse_loss(pred, target)


def main():
    key_embs, finger_emb = creat_key_finger_emb()
    in_dims = key_embs.shape[1] + finger_emb.shape[1]
    model = Motion(in_dims=in_dims, out_dims=position_dims, time_window=args.time_window).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss().to(device)
    T = 100
    batch_size = 32
    EPOCHS = 200
    with open(args.log_path, 'a+') as f:
        argsDict = args.__dict__
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------'+ '\n')
        for epoch in range(EPOCHS):
            # train
            for step in tqdm.tqdm(range(T)):
                key_idxs = np.random.choice(key_num, size=batch_size)
                finger_idxs = np.random.choice(finger_num, size=batch_size)    
                labels = np.clip(key_idxs - finger_idxs, a_min=0, a_max=position_num-1)
                in_emb = np.concatenate([key_embs[key_idxs], finger_emb[finger_idxs]], axis=-1)
                x = torch.from_numpy(in_emb).to(device)
                y = torch.from_numpy(TARGETS_FAC[labels]).to(device)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # test
            
            loss_record = []
            f.writelines('\n')
            f.writelines('Epoch:[{epoch}/{total_eps}]:\n'.format(epoch=epoch, total_eps=EPOCHS))
            for key in range(key_num):
                for fin in range(finger_num):
                    in_emb = np.concatenate([key_embs[key], finger_emb[fin]], axis=-1)
                    x = torch.from_numpy(in_emb).to(device)
                    with torch.no_grad():
                        pred = model(x)
                    
                    y = max(min(key - fin, position_num-1), 0)
                    target = torch.from_numpy(TARGETS_FAC[y]).to(device)
                    loss = F.mse_loss(pred, target, reduction='sum')
                    loss_record.append(loss.cpu().item())
                    real_pred = pred.cpu().numpy() / t_factors
                    distant = np.sum((TARGETS[y] - real_pred)**2)**0.5
                    f.writelines('  Predict position {y}: {pred}\n'.format(y=y, pred=real_pred.tolist()))
            f.writelines('==> Epoch:[{epoch}/{total_eps}][validation stage]: loss: {loss}, distant {dis}\n'.format(
                    epoch=epoch, total_eps=EPOCHS, loss=sum(loss_record)/len(loss_record), dis=distant))
            print('==> Epoch:[{epoch}/{total_eps}][validation stage]: loss: {loss}, distant {dis}\n'.format(
                    epoch=epoch, total_eps=EPOCHS, loss=sum(loss_record)/len(loss_record), dis=distant))
            

if __name__ == '__main__':
    main()