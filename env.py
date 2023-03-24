from cgi import test
import torch
import numpy as np
import math
import itertools
import matplotlib.pyplot as plt
from operator import itemgetter
import operator
import random
from itertools import chain

def dist_eval(ids1,ids2,dimX,dimY,dimZ):
    t1 = id2coor(ids1,dimX,dimY,dimZ)
    t2 = id2coor(ids2,dimX,dimY,dimZ)
    dist = abs(t1[:,0]-t2[:,0]) + abs(t1[:,1]-t2[:,1])
    return dist

def node_nebrs(id,dimX,dimY):
    id_crds = id2coor_batch(id,dimX,dimY,2)
    nebrs   = torch.zeros(4*id.shape[0],3)
    i = 0
    nebrs[:i+id.shape[0]][:,0],nebrs[:i+id.shape[0]][:,1] = id_crds[:,0]+1,id_crds[:,1]
    i = i+id.shape[0]
    nebrs[i:i+id.shape[0]][:,0],nebrs[i:i+id.shape[0]][:,1] = id_crds[:,0]-1,id_crds[:,1]
    i = i+id.shape[0]
    nebrs[i:i+id.shape[0]][:,1],nebrs[i:i+id.shape[0]][:,0] = id_crds[:,1]+1,id_crds[:,0]
    i = i+id.shape[0]
    nebrs[i:i+id.shape[0]][:,1],nebrs[i:i+id.shape[0]][:,0] = id_crds[:,1]-1,id_crds[:,0]
    nebrs[nebrs==dimX] = dimX-1
    nebrs[nebrs==-1] = 1
    return coor2id(nebrs.to('cuda'),dimX,dimY,2)



def hop_cnt(TSV_id,src,dst,nw_dict,dimX,dimY,dimZ):  # Evaluates no of hops between src and dst
    r1 = id2coor(src,dimX,dimY,dimZ)
    r2 = id2coor(dst,dimX,dimY,dimZ)
    hop_dst = abs(r1[:,0]-r2[:,0])+abs(r1[:,1]-r2[:,1])
    hop_dst = torch.where((r1[:,2] == r2[:,2]),hop_dst.to(float),0.).to(dtype=torch.int32)
    d = torch.where(hop_dst==0)
    
    Vlink_dsts = torch.zeros(d[0].shape[0],TSV_id.shape[1]) 
    for pos in range(TSV_id.shape[1]):
        Vl_cor = id2coor(TSV_id[d][:,pos],dimX,dimY,dimZ)
        Vlink_dsts[:,pos]= abs(r1[d][:,0]-Vl_cor[:,0])+abs(r1[d][:,1]-Vl_cor[:,1])
    mlocs = torch.min(Vlink_dsts,1)
    a = mlocs[1].to(device='cuda')
    b = torch.unsqueeze(a,1)
    Vlinks_batch = torch.gather(TSV_id[d],1,b)
    if(Vlinks_batch.shape[0] == 0):
        return hop_dst
    if(Vlinks_batch.shape[0] > 1):
        Vlinks_batch = torch.squeeze(Vlinks_batch)
    vl_cords = id2coor(Vlinks_batch,dimX,dimY,dimZ)
    min_dists = mlocs[0].to(dtype=torch.int).to(device='cuda')
    hop_dst[d] = min_dists + abs(vl_cords[:,0]-r2[d][:,0])+abs(vl_cords[:,1]-r2[d][:,1])+1
    return hop_dst

def TSV_loc(tsv_info,nw_dict,dimX,dimY):
    TSV_ids = {}
    for loc,val in enumerate(tsv_info[0]): 	
        if(val == 1):
            TSV_ids[str(loc)] = {'cor':nw_dict[str(loc)]['cor']}
            dst_vlk = loc+(dimX * dimY)
            TSV_ids[str(dst_vlk)] = {'cor':nw_dict[str(dst_vlk)]['cor']}
    return TSV_ids

def coor2id(tsv_cords,dimX,dimY,dimZ):
    ids = tsv_cords[:,2]*dimX*dimY + tsv_cords[:,1]*dimX + tsv_cords[:,0]
    
    return ids

def id2coor(id,dimX,dimY,dimZ):
    coord = torch.zeros(len(id),3).int().to(device='cuda')
    coord[:,2] = id / (dimX*dimY)  # z cord
    coord[:,1] = (id-coord[:,2]*dimX*dimY) /dimX
    coord[:,0] = (id-coord[:,2]*dimX*dimY) % dimX
    return coord

def id2coor_batch(id,dimX,dimY,dimZ):
    coord = torch.zeros(id.shape[0],3).int().to(device='cuda')
    # coord[:,2] = id / (dimX*dimY)  # z cord
    coord[:,1] = (id-coord[:,2]*dimX*dimY) /dimX
    coord[:,0] = (id-coord[:,2]*dimX*dimY) % dimX
    return coord


class Env_tsp():
    def __init__(self, cfg,nw_dict,traff_dict,traff_data):
        '''
        nodes(cities) : contains nodes and their 2 dimensional coordinates 
        [city_t, 2] = [3,2] dimension array e.g. [[0.5,0.7],[0.2,0.3],[0.4,0.1]]
        '''
        self.batch  = cfg.num_samples
        # self.max_iter = cfg.max_iter
        
        self.dimX   = cfg.dimX
        self.dimY   = cfg.dimY
        self.dimZ   = cfg.dimZ
        self.city_t = self.dimX*self.dimY
        self.traff_dict = traff_dict
        self.nw_dict = nw_dict
        self.traff_data = traff_data
      

#%%
    def test_cost2(self,mapp,tsv):
        cost=torch.zeros(mapp.shape[0]).to(device='cuda')
        for i in list(self.traff_dict.keys()):
            p1 = (mapp == self.traff_dict[i]['src']).nonzero(as_tuple=True)[1]
            p2 = (mapp == self.traff_dict[i]['dst']).nonzero(as_tuple=True)[1]
            hops = hop_cnt(tsv,p1,p2,self.nw_dict,self.dimX,self.dimY,self.dimZ)
            cost += self.traff_dict[i]['val']*hops
        return cost
    
    def test_cost(self,mapp):
        cost=torch.zeros(mapp.shape[0])
        ml = self.dimX*self.dimY*self.dimZ
        for i in list(self.traff_dict.keys()):
            p1 = (mapp[0:ml] == self.traff_dict[i]['src']).nonzero(as_tuple=True)[0]
            p2 = (mapp[0:ml] == self.traff_dict[i]['dst']).nonzero(as_tuple=True)[0]
            hops = hop_cnt(mapp[ml:len(mapp)],p1,p2,self.nw_dict,self.dimX,self.dimY)
            cost += self.traff_dict[i]['val']*hops
        return cost
#%%
    def show(self, nodes, tour,tsv):
        tsv_cords1 = id2coor(tsv,self.dimX,self.dimY,self.dimZ)
        nodes = nodes.cpu().detach()
        tour  = tour[:].cpu().detach()
        tsv_cords1   = tsv_cords1[:].cpu().detach()
        b= torch.stack((tsv_cords1,tsv_cords1),dim=1)
        plt.figure(figsize=(19.20,10.80))
        ax = plt.axes(projection='3d')
        ax.plot3D(nodes[:, 0], nodes[:, 1],nodes[:,2], 'yo', markersize=25)
        for k in range(tsv.shape[0]):
            ax.plot3D([b[k][0][0],b[k][1][0]],[b[k][0][1],b[k][1][1]],[0,1],'r-',linewidth=3)
        i = 0
        for x, y, z in nodes:
            label = '%d' % (tour[i])   
            i = i+1
            ax.text(x, y, z, label)
        i = 0
        for x, y, z in nodes:
            label = '(%d,%d,%d)' % (x,y,z)   
            i = i+1
            ax.text(x-0.1, y-0.1, z-0.1, label)

        for z in range(self.dimZ):
            for y in range(self.dimY):
                for x in range(self.dimX-1):
                    ax.plot3D([x+0.1,x+1-0.1],[y,y],[z,z],'k-')

        for z in range(self.dimZ):
            for x in range(self.dimX):
                for y in range(self.dimY-1):
                    ax.plot3D([x,x],[y+0.2,y+1-0.1],[z,z],'k-')
        ax.view_init(elev=22, azim=-55)
        ax.set_xlabel('$X$', fontsize=20)
        ax.set_ylabel('$Y$',fontsize=20)
        ax.set_zlabel('$Z$', fontsize=20)        
        plt.savefig('plots/mesh3D_'+str(self.dimX)+str(self.dimY)+str(self.dimZ)+'.eps', format='eps',dpi=1200,bbox_inches='tight')
        # plt.show()
        
   
    def shuffle(self, inputs):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        inputs = torch.squeeze(inputs)
        shuffle_inputs = torch.zeros(inputs.size())

        for i in range(self.batch):
            perm = torch.randperm(self.city_t)
            shuffle_inputs[i,:,:] = inputs[i,perm,:]
        return shuffle_inputs

    def back_tours(self, pred_shuffle_tours, shuffle_inputs, test_inputs, device):
        '''
        pred_shuffle_tours:(batch,city_t)
        shuffle_inputs:(batch,city_t_t,2)
        test_inputs:(batch,city_t,2)
        return pred_tours:(batch,city_t)
        '''
        test_inputs = torch.squeeze(test_inputs)
        # test_inputs = test_inputs[:,:,None]
        pred_tours = []
        for i in range(self.batch):
            pred_tour = []
            for j in range(int(self.city_t/4)):
                xy_temp = shuffle_inputs[i,
                                         pred_shuffle_tours[i, j]].to(device)
                for k in range(self.city_t):
                    if torch.all(torch.eq(xy_temp, test_inputs[i, k])):
                        pred_tour.append(torch.tensor(k))
                        # if len(pred_tour) == self.city_t:
                        pred_tours.append(torch.stack(pred_tour, dim=0))
                        break
        pred_tours = torch.stack(pred_tours, dim=0)
        return pred_tours

    def get_random_tsv(self):
        tour = []
        while len(tour) < int(self.city_t/4):
            city = np.random.randint(self.city_t)
            if city not in tour:
                tour.append(city)
        tour = torch.from_numpy(np.array(tour))
        return tour

    def stack_random_tsvs(self):
        list = [self.get_random_tsv() for i in range(self.batch)]
        tours = torch.stack(list, dim = 0)
        return tours

# %%
    def correct_TSVs(self,tsv_ip):
        ln = tsv_ip.shape[1]
        dim1 = int(ln*(ln-1)/2)
        valid_tsvs = torch.zeros(tsv_ip.shape[0],dim1).to(int)
        tsv_pairs = torch.zeros(tsv_ip.shape[0],dim1,2)
        mod_tsp = torch.zeros(tsv_ip.shape).to(int).to('cuda')
        # create a list containing tsv pairs
        mod_tsp[:] = tsv_ip[:]
        for rep in range(2):
            k = 0
            for i in range(ln):
                j=i+1
                while j<ln:
                    valid_tsvs[:,k] = dist_eval(mod_tsp[:,i],mod_tsp[:,j],self.dimX,self.dimY,self.dimZ)
                    tsv_pairs[:,k][:,0], tsv_pairs[:,k][:,1]   =  mod_tsp[:,i],mod_tsp[:,j]
                    j = j+1
                    k = k+1
            a = (valid_tsvs == 1).nonzero(as_tuple=True)[0]
            a = torch.unique(a)
            # c = node_nebrs(tsv_ip[a],self.dimX,self.dimY)
            for tnum in a:
                tot_ids = []
                # mod_tsp[tnum] = tsv_ip[tnum]
                id_nbrs = node_nebrs(tsv_ip[tnum],self.dimX,self.dimY)
                pos = (valid_tsvs[tnum]==1).nonzero(as_tuple=True)[0][0]
                pos1    = (tsv_ip[tnum]== tsv_pairs[tnum][pos][0]).nonzero(as_tuple=True)[0][0]
                # a = (tsv_ip[tnum]>(self.city_t/2)).sum()
                # flat_ids = torch.tensor(list(chain.from_iterable(tot_ids)))
                # if(a>= ln/2):
                    #     # ch_tloc = torch.randint(0,int(self.city_t/2),(1,))
                        
                not_in_S = random.choice([x for x in range(self.city_t) if x not in id_nbrs])
                    # else:
                    #     not_in_S = random.choice([x for x in range(int(self.city_t/2),self.city_t) if x not in flat_ids])  

                mod_tsp[tnum][pos1]  =  not_in_S

            
        return mod_tsp.to('cuda') 

    def tsv_repair(self,tsvs):
        correct_tsvs = torch.zeros(tsvs.shape).to(int).to('cuda')
        for iter in range(tsvs.shape[1]):
            tsv_cords = id2coor(tsvs[:,iter],self.dimX,self.dimY,self.dimZ)
            tsv_cords[:,2] = 0
            correct_tsvs[:,iter] = coor2id(tsv_cords,self.dimX,self.dimY,self.dimZ)
            
        
        return correct_tsvs