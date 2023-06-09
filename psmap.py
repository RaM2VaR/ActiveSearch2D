#%%
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers
import torch
import math
from utils.utils import communication_cost_multiple_samples,LPNet_pred,booksim_latency,communication_energy_multiple_samples
from utils.datagenerate import generate_distance_matrix
from torch_geometric.loader import DataLoader
from numba import njit
from models.mpnn_ptr import MpnnPtr
from timeit import default_timer as timer
import random
import argparse
import sys
from graphdataset import get_transform
#%%
@njit
def calc_swap_seq(curr_prtl, trns_to_prtl):
    """
    Calculate the swap sequence to transform curr_prtl to trns_to_prtl 
    """
    # copy curr_prtl
    curr_prtl_cpy = np.copy(curr_prtl)
    seq = np.zeros((len(curr_prtl-1),), dtype=np.int32 ) 
    for i in range(curr_prtl_cpy.shape[0] -1):
        item = trns_to_prtl[i]
        index = 0
        for j in range(curr_prtl_cpy.shape[0]):
            if curr_prtl_cpy[j] == item:
                index = j
                break
        seq[i] = index
        curr_prtl_cpy[i], curr_prtl_cpy[index] = curr_prtl_cpy[index], curr_prtl_cpy[i]  
    return seq
@njit
def rand_permute2D(particle):
    for i in range(particle.shape[0]):
        particle[i] = np.random.permutation(particle.shape[1])
        # print(particle[i])
@njit
def apply_swap_seq(particle, seq, p=1):
    for i in range(seq.size):
        if random.random() > p:
            particle[seq[i]], particle[i] = particle[i], particle[seq[i]]

@njit
def evolve_particles(prtls, lcl_bst_prtl, gbest):
    """
    Evolves particles using lcl_locs for lcl_bst_prtl and gbest_locs for gbest
    """
    for i in range(prtls.shape[0]):
        curr_prtl = prtls[i]
        curr_lcl_bst_prtl = lcl_bst_prtl[i]
        seq_lcl = calc_swap_seq(curr_prtl, curr_lcl_bst_prtl)
        seq_gbl = calc_swap_seq(curr_prtl, gbest)
        apply_swap_seq(prtls[i], seq_lcl, .5)
        apply_swap_seq(prtls[i], seq_gbl, .5)        
#%%
class Psmap:
    def __init__(self, dataset_path, num_particles,\
        max_iterations, prtl_init_method="random", model_path=None, max_graph_size=121):
        device = torch.device("cpu")
        single_graph_data = torch.load(dataset_path, map_location=device)
        single_graph_data = single_graph_data[0]  # comment this line for taffic other than bench mark
        self.max_graph_size = max_graph_size
        self.transform = get_transform(max_graph_size)
        single_graph_data = self.transform(single_graph_data)
        graph_size = single_graph_data.num_nodes
        n = math.ceil(math.sqrt(graph_size))
        m = math.ceil(graph_size/n)
        datalist = [single_graph_data]
        # just for using communication_cost function
        dataloader = DataLoader(datalist, batch_size=1)
        self.data = next(iter(dataloader))
        self.distance_matrix = generate_distance_matrix(n,m).to(device)
        self.max_iterations = max_iterations
        self.num_particles = num_particles
        self.particle_size = graph_size
        self.prtl_init_method = prtl_init_method
        self.model_path = model_path
        self.transform = get_transform()
        self.device = device
    def comm_cost(self, particle):
        if args.obj_fun == 'LPNet':
            penalty = LPNet_pred(torch.tensor(particle),self.particle_size,args.inj_rate,args.dataset)
            # penalty = torch.tensor(penalty)
        elif args.obj_fun == 'sim_lat':
            penalty = booksim_latency(torch.tensor(particle),self.particle_size,args.inj_rate,args.dataset)
        elif args.obj_fun == 'comm_cost':
            penalty = communication_cost_multiple_samples(self.data.edge_index,self.data.edge_attr,
                            self.data.batch, self.distance_matrix, torch.from_numpy(particle), particle.shape[0]).detach().numpy()
        elif args.obj_fun == 'comm_energy':
            penalty = communication_energy_multiple_samples(self.data.edge_index,self.data.edge_attr,
                            self.data.batch, self.distance_matrix, torch.from_numpy(particle), particle.shape[0]).detach().numpy()
        else:
            raise ValueError('Obj function is undefined')
        return penalty
    
    def prtl_init(self, num_particles):
        particle = np.zeros((num_particles, self.particle_size), dtype=np.int64)
        rand_permute2D(particle)
        return particle
    def prtl_init_model(self, num_particles):
        # load model
        # TODO: generate half population from model and half randomly
        num_sampled_particles = int(num_particles * .8)
        num_random_particles = num_particles - num_sampled_particles
        max_graph_size = self.max_graph_size
        mpnn_ptr = MpnnPtr(input_dim=max_graph_size, embedding_dim=max_graph_size + 7,
            hidden_dim=max_graph_size + 7, K=3, n_layers=1, p_dropout=0, device=self.device, logit_clipping=False)
        mpnn_ptr.decoding_type = "greedy"
        if self.model_path is None:
            raise ValueError('model_path is None')
        mpnn_ptr.load_state_dict(torch.load(self.model_path, map_location=self.device))
        mpnn_ptr.eval()
        with torch.no_grad():
            # generate initial population
            with torch.no_grad():
                particle_sampled, _ = mpnn_ptr(self.data, 3000)
        particle_sampled = particle_sampled.detach().numpy()
        particle_sampled_fit = self.comm_cost(particle_sampled)
        indices = particle_sampled_fit.argpartition(num_sampled_particles)[:num_sampled_particles]
        particle_first_half = particle_sampled[indices]
        # np.set_printoptions(threshold=sys.maxsize)
        # print(f'{np.unique(particle_first_half,axis=0)}')
        particle_second_half = self.prtl_init(num_random_particles)
        particle = np.concatenate((particle_first_half, particle_second_half), axis=0)
        return particle
    def global_best(self, particle, prtl_fitness):
        min_id = np.argmin(prtl_fitness)
        # print(prtl_fitness)
        # print(min_id)
        gbest = particle[min_id]
        gbest_fit = prtl_fitness[min_id]
        return gbest, gbest_fit
    
    def run(self):
        if self.prtl_init_method == "random":
            prtl = self.prtl_init(self.num_particles)
        elif self.prtl_init_method == "model":
            prtl = self.prtl_init_model(self.num_particles)
        else:
            raise ValueError(f'{self.prtl_init_method} is not supported for particle initialization')
        prtl_fit = self.comm_cost(prtl)
        lcl_bst_prtl = np.copy(prtl)
        lcl_bst_fit = np.copy(prtl_fit)
        gbest, gbfit = self.global_best(prtl, prtl_fit)
        gb_fits = [gbfit]
        gb_prtls = [gbest]
        # 2nd generation randomly shuffle contents of the prtl
        np.random.shuffle(prtl)
        prtl_fit = self.comm_cost(prtl)
        update_condition = prtl_fit < lcl_bst_fit
        lcl_bst_prtl = \
                np.where(update_condition.reshape(self.num_particles, 1), prtl, lcl_bst_prtl)
        lcl_bst_fit = np.where(update_condition, prtl_fit, lcl_bst_fit)
        gbest_t, gbfit_t = self.global_best(prtl, prtl_fit)
        if gbfit > gbfit_t:
            gbest = gbest_t
            gbfit = gbfit_t
            gb_fits.append(gbfit)
            gb_prtls.append(gbest)
        
        check = 0
        for j in range(self.max_iterations):
            # next generation of good particles
            evolve_particles(prtl, lcl_bst_prtl, gbest)
            # calculate global and local best for this generation
            prtl_fit = self.comm_cost(prtl)
            # update lcl_bst_prtl and lcl_bst_fit
            update_condition = prtl_fit < lcl_bst_fit
            lcl_bst_prtl = \
                np.where(update_condition.reshape(self.num_particles, 1), prtl, lcl_bst_prtl)
            lcl_bst_fit = np.where(update_condition, prtl_fit, lcl_bst_fit)
            # update gbest and gbfit for this generation
            gbest_t, gbfit_t = self.global_best(prtl, prtl_fit)
            if gbfit > gbfit_t:
                gbest = gbest_t
                gbfit = gbfit_t
                gb_fits.append(gbfit)
                gb_prtls.append(gbest)
                check = 0
            else:
                check = check +1
            if check > 50:
                break
            if (j % 10 == 0):
                print(f'iteration {j}: gbest fitness {gbfit}')
        gb_fits = np.array(gb_fits)
        min_fit_idx = np.argmin(gb_fits)
        return gb_prtls[min_fit_idx], gb_fits[min_fit_idx]
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help="path to the data file", type=str)
    parser.add_argument('-p','--num_prtl', help="number of particles in each generation", type=int, default=100)
    parser.add_argument('-i','--max_iter', help="max number of iterations", type=int, default=1000)
    parser.add_argument('-m','--model', help="path to the model file for initial population generation", type=str, default=None)
    parser.add_argument('--obj_fun',help='objective function (comm_cost/LPNet/comm_energy)',type=str,default='comm_cost')
    parser.add_argument('--inj_rate',help='injection rate for LPNet model',type=float,default=0.001)
    args = parser.parse_args()
    prtl_init_method = "random" if args.model is None else "model"
    if args.obj_fun == 'LPNet':
        print('-----------fitness function is packet latency (LPNet)------')
    elif args.obj_fun =='comm_cost':
        print('-----------fitness function is communication cost------')
    elif args.obj_fun == 'comm_energy':
        print('-----------fitness function is communication Energy------')
    elif args.obj_fun == 'sim_lat':
        print('*********** fitness function is latency(simulator) ************')
    else:
        raise ValueError('Fitness function undefined')
    psmap = Psmap(args.dataset, args.num_prtl, args.max_iter, prtl_init_method, args.model)
    # best_prtl,cost = psmap.run()
 
    # measure time taken and best cost by running the algorithm 5 times
    start_time = timer()
    best_prtl, cost = psmap.run()
    end_time = timer()
    print(f'best particle {best_prtl}')
    print(f'best cost {cost:.3f}')
    time_taken = end_time - start_time
    # data_name = args.dataset.split('/')[-1].split('.')[0] b
    bst_mapp = best_prtl.tolist()
    file1 = open("results/sim_results/PSMAP_"+args.obj_fun+args.dataset[-16:-3]+'.csv','a')
    file1.write(f'{args.dataset[-16:-3]},{args.inj_rate},{cost:.2f} ,{time_taken:.2f},{bst_mapp}\n')
# python3 psmap.py data_tgff/data_single_TGFF1_16.pt --num_prtl 1024 --max_iter 5000 --model models_data_final/model_16_01-10.pt
# python psmap.py data_tgff/final_before_mtp/data_single_TGFF1_norm_16.pt --num_prtl 500 --max_iter 1000 --model models_data_multiple/small/models_data/model_init_pop_04-21_16-10.pt