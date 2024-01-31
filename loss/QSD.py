import math 
import torch
import numpy as np
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import torch.cuda.comm as comm
from torch.distributed import ReduceOp


class QSD_loss(nn.Module):
    def __init__(self, distance = 'L2'):
        super(QSD_loss, self).__init__()
        if distance == 'L2':
            self.loss_layer = nn.MSELoss()
        elif distance == 'L1':
            self.loss_layer = nn.L1Loss()
        else:
            print("can not parsing prase")
        self.transform = nn.ReLU()
        self.kl_loss=nn.KLDivLoss().cuda()
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
        
    def forward(self, features_1, features_2, quality_1, quality_2, weights):
        losses = []
        thres = 0.3
        quality_mask = abs(quality_1-quality_2)
        mean_quality_dis = np.mean(quality_mask)
        margin_upper = 100 - (100-mean_quality_dis)*thres
        margin_lower = mean_quality_dis*thres
        mask = (quality_mask < margin_lower) | (quality_mask > margin_upper)
        quality_1[mask] = 0
        quality_2[mask] = 0
        quality_mask = torch.from_numpy(quality_mask).cuda()
        fea1_high_index = np.argwhere(quality_1>quality_2).squeeze()
        fea2_high_index = np.argwhere(quality_1<quality_2).squeeze()
        fea1_high_index = torch.from_numpy(fea1_high_index).cuda()
        fea2_high_index = torch.from_numpy(fea2_high_index).cuda()
        if sum(quality_1) == 0: return torch.tensor(0), torch.tensor([0,0,0,0,0,0])

        for fea_1, fea_2 in zip(features_1, features_2):
            fea_t1 = torch.index_select(fea_1, 0, fea1_high_index)
            fea_s1 = torch.index_select(fea_1, 0, fea2_high_index)
            fea_t2 = torch.index_select(fea_2, 0, fea2_high_index)
            fea_s2 = torch.index_select(fea_2, 0, fea1_high_index)
            fea_t = torch.cat([fea_t1, fea_t2], 0)
            fea_s = torch.cat([fea_s2, fea_s1], 0)
            fea_t_detached = fea_t.clone().detach()
            if len(fea_t_detached.size())==2:
                cos_dis = 1 - self.cos(fea_t_detached, fea_s) 
                loss =  torch.mean(cos_dis)
                losses.append(loss)
            else:
                fea_s = torch.mean(torch.pow(fea_s, 2), dim = 1)
                fea_t_detached = torch.mean(torch.pow(fea_t_detached, 2), dim = 1)
                mse_loss = self.loss_layer(fea_s, fea_t_detached)
                ampify = 2/ (torch.max(fea_s) - torch.min(fea_s))
                losses.append(ampify * mse_loss)
        
        loss_all = 0
        weighted_losses = []
        for i in range(len(losses)):
            loss_all = loss_all + losses[i]*weights[i]
            weighted_losses.append(losses[i]*weights[i])
        return loss_all, weighted_losses 
            
if __name__ == "__main__":
    import numpy as np
    loss = QSD_loss(distance = 'L2')
    quality_1 = np.asarray([30.1, 50.2, 70.7, 90.1])
    quality_2 = np.asarray([35.2, 45.3, 47.5, 92.5])
    feats_1 = []
    feats_2 = []
    weights = [1, 1, 1, 1, 0.1]
    for i in range(5):
        feats_1.append((i+1)*torch.ones(4, (i+1)*4, 64, 64).cuda())
        feats_2.append(torch.zeros(4, (i+1)*4, 64, 64).cuda())
    loss_val = loss(feats_1, feats_2, quality_1, quality_2, weights)
    print(loss_val) 
        