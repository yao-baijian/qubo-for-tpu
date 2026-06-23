import torch
import numpy as np
import math
import torch.nn.functional as Fun
import os 
import time 
import re

def read_cnf(path):
    print('Reading file: '+ path)
    with open(path,'r') as f:
        lines = f.readlines()
    k_length_sat_table = {}
    for line in lines:
        l = line.split()
        if l[0] == 'c':  # comment line
            pass
        elif l[0] == 'p': # problem line
            N, M = map(int, l[2:])
        else:
            clause = list(map(int, l[:-1]))
            k = len(clause)
            if k not in k_length_sat_table:
                k_length_sat_table[k] = []
            k_length_sat_table[k].append([])
            k_length_sat_table[k][-1].append(list(map(abs, clause)))
            q_states = []
            for i in range(k):
                if clause[i] > 0:
                    q_states.append(0)    # postive literal
                else:
                    q_states.append(1)     # negative literal
            k_length_sat_table[k][-1].append(q_states)
    sat_table = []
    minimum_index = []
    maximum_index = []
    for key in sorted(k_length_sat_table.keys()):
        k_length_sat_table[key] = np.array(k_length_sat_table[key])
        minimum_index.append(np.min(k_length_sat_table[key][:,0,:]))
        maximum_index.append(np.max(k_length_sat_table[key][:,0,:]))
    max_idx = max(maximum_index)
    min_idx = min(minimum_index)
    assert max_idx - min_idx + 1  == N
    for key in sorted(k_length_sat_table.keys()):
        k_length_sat_table[key][:,0,:] -= min_idx
        k_length_sat_table[key] = k_length_sat_table[key].tolist()
        sat_table += k_length_sat_table[key]
    real_M = len(sat_table)
    assert real_M  == M
    max_k = max(k_length_sat_table.keys())
    min_k = min(k_length_sat_table.keys())
    M_k = {}
    for key in sorted(k_length_sat_table.keys()):
        M_k[key] = len(k_length_sat_table[key])
    return N, M, M_k, max_k, min_k, sat_table


def clause_mask_tensor(N, M, M_k, sat_table, batch, q=2):
    clause = [[] for xx in range(len(M_k))]
    clause_batch = [[] for xx in range(len(M_k))]
    keys = [xx for xx in sorted(M_k.keys())]
    for ii in range(M):
        k = len(sat_table[ii][0])
        clause_m = torch.sparse_coo_tensor([sat_table[ii][0],
                                 sat_table[ii][1]], 
                                    [1] * k,(N, q)).to_dense().unsqueeze(0).repeat(batch,1,1).to_sparse()
        clause[keys.index(k)].append(clause_m.unsqueeze(0))
    for kk in keys:
        clause_batch[keys.index(kk)] = torch.cat(clause[keys.index(kk)],dim=0)
    # clause_batch = torch.cat(clause,dim=0)    # [M, batch, N, q]  # sparse tensor values = k * M * batch
    return clause_batch 
    
class FEM_MaxSAT:
    def __init__(
            self, J_m, N, M_k, max_k, min_k, beta, learning_rate, replicas, c_grad, wd, alpha, dampen, mom, dev = 'cuda:0', dtype=torch.float32,
            seed = None, q = 2, h_factor = 0.001):
        self.dtype = dtype
        self.dev = dev
        self.J_m = [J_m[ii].to(self.dtype).to(self.dev) for ii in range(len(J_m))]
        self.beta = torch.from_numpy(beta).to(self.dtype).to(self.dev) 
        self.N, self.M_k, self.max_k, self.min_k = N, M_k, max_k, min_k
        self.keys = [xx for xx in sorted(M_k.keys())]
        self.replicas = replicas
        self.learning_rate = learning_rate
        self.seed = seed
        self.wd,self.alpha,self.mom = wd, alpha, mom
        self.dampen = dampen
        self.q = q
        self.c_grad = c_grad
        self.h_factor = h_factor
        self.initialize()
        
    def initialize(self):
        if self.seed is not None:
            torch.manual_seed(self.seed)
        self.h = self.h_factor * torch.randn([1, self.replicas, self.N, self.q], device=self.dev, dtype=self.dtype)
        self.optimizer = torch.optim.RMSprop([self.h], lr=self.learning_rate,weight_decay=self.wd,momentum=self.mom,alpha=self.alpha)
    def manual_grad_maxksat(self, step):
        minus_p = 1 - 0.99999999 * self.p
        grads = []
        for kk in self.keys:
            prod = self.J_m[self.keys.index(kk)] * minus_p
            value = prod._values().view(self.M_k[kk], self.replicas, kk)
            value_prod = value.prod(-1, keepdim=True)
            grads.append(torch.sparse_coo_tensor(prod._indices(),
                                (-value_prod/value).reshape(-1), 
                                prod.shape))
        grad_M = torch.cat(grads,dim=0)
        grad = grad_M.sum(0,keepdim=True).to_dense()
        tp = self.c_grad * (grad + (torch.log(self.p+1e-30))/self.beta[step])
        h_grad = (tp  - (tp * self.p).sum(3,keepdim=True).expand(tp.shape)) * self.p
        
        return h_grad

    def iterate(self):
        for step in range(self.beta.shape[0]):
            """ unnormalized probabilities as a function of field self.h """
            self.p = torch.softmax(self.h, dim=3)  
            self.optimizer.zero_grad()
            self.h.grad = self.manual_grad_maxksat(step)
            self.optimizer.step()
    
    def cal_energy(self):
        config = Fun.one_hot(self.p.argmax(dim=3),num_classes=self.p.shape[3])
        minus_p = 1 - config
        energy = 0 
        for kk in self.keys:
            prod = self.J_m[self.keys.index(kk)] * minus_p
            value = prod._values().reshape(self.M_k[kk], self.replicas, kk)   # # values = k * M * batch
            value_prod = value.prod(-1,keepdim=True)
            energy += value_prod.sum(0).reshape(-1)
        return energy
    
def main(instance, device):
    if '.cnf' in instance:
        instance = re.sub('.cnf', '', instance)
    path = "./ms_random_combine/" +instance+ ".cnf"
    N, M,  M_k, max_k, min_k, sat_table = read_cnf(path)
    params = load_params(instance)
    J_m = clause_mask_tensor(N, M, M_k, sat_table, params['replicas'], 2)
    dtype = torch.float32
    seed = np.random.choice(np.arange(500))  # set any integer if seed is specified.
    # seed = 60
    print('seed: ',seed)
    
    beta = 1/np.linspace(params['T_max'],params['T_min'],params['N_steps'])
    fem = FEM_MaxSAT(J_m, N, M_k, max_k, min_k, beta, params['lr'], params['replicas'], params['c_grad'], params['wd'], 
                         params['alpha'], dampen=None, mom=params['mom'], dev = device, dtype=dtype,
            seed = seed, q = 2, h_factor = 0.001)
    start_t = time.perf_counter()
    fem.iterate()
    end_t = time.perf_counter()
    energy = fem.cal_energy()
    print(f"minimum found by gradmfa: {energy.min()}, \nelapsed time: {end_t - start_t} seconds.")
    print('mean:', energy.mean().item())

   
def load_params(instance):
    if instance == "s2v120c1200-1":
        params={"ins":"s2v120c1200-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v120c1200-2":
        params={"ins":"s2v120c1200-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v120c1200-3":
        params={"ins":"s2v120c1200-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v120c1300-1":
        params={"ins":"s2v120c1300-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v120c1300-2":
        params={"ins":"s2v120c1300-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v120c1300-3":
        params={"ins":"s2v120c1300-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v120c1400-1":
        params={"ins":"s2v120c1400-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v120c1400-2":
        params={"ins":"s2v120c1400-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v120c1400-3":
        params={"ins":"s2v120c1400-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v120c1500-1":
        params={"ins":"s2v120c1500-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v120c1500-2":
        params={"ins":"s2v120c1500-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v120c1500-3":
        params={"ins":"s2v120c1500-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v120c1600-1":
        params={"ins":"s2v120c1600-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v120c1600-2":
        params={"ins":"s2v120c1600-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v120c1600-3":
        params={"ins":"s2v120c1600-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v120c1700-1":
        params={"ins":"s2v120c1700-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v120c1700-2":
        params={"ins":"s2v120c1700-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v120c1700-3":
        params={"ins":"s2v120c1700-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v120c1800-1":
        params={"ins":"s2v120c1800-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v120c1800-2":
        params={"ins":"s2v120c1800-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v120c1800-3":
        params={"ins":"s2v120c1800-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v120c1900-1":
        params={"ins":"s2v120c1900-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v120c1900-2":
        params={"ins":"s2v120c1900-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v120c1900-3":
        params={"ins":"s2v120c1900-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v120c2000-1":
        params={"ins":"s2v120c2000-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v120c2000-2":
        params={"ins":"s2v120c2000-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v120c2000-3":
        params={"ins":"s2v120c2000-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v120c2100-1":
        params={"ins":"s2v120c2100-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v120c2100-2":
        params={"ins":"s2v120c2100-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v120c2100-3":
        params={"ins":"s2v120c2100-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v120c2200-1":
        params={"ins":"s2v120c2200-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v120c2200-2":
        params={"ins":"s2v120c2200-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v120c2200-3":
        params={"ins":"s2v120c2200-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v120c2300-1":
        params={"ins":"s2v120c2300-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v120c2300-2":
        params={"ins":"s2v120c2300-2","N_steps":60,"replicas":50,"T_min":0.0001,"T_max":0.47,"c_grad":1.46,"lr":1.05,"alpha":0.33,"wd":0.028,"mom":0.87,"optimizer":"RMSprop"}
    elif instance == "s2v120c2300-3":
        params={"ins":"s2v120c2300-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v120c2400-1":
        params={"ins":"s2v120c2400-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v120c2400-2":
        params={"ins":"s2v120c2400-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v120c2400-3":
        params={"ins":"s2v120c2400-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v120c2500-1":
        params={"ins":"s2v120c2500-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v120c2500-2":
        params={"ins":"s2v120c2500-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v120c2500-3":
        params={"ins":"s2v120c2500-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v120c2600-1":
        params={"ins":"s2v120c2600-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v120c2600-2":
        params={"ins":"s2v120c2600-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v120c2600-3":
        params={"ins":"s2v120c2600-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v140c1200-1":
        params={"ins":"s2v140c1200-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v140c1200-2":
        params={"ins":"s2v140c1200-2","N_steps":100,"replicas":50,"T_min":0.0001,"T_max":0.62,"c_grad":0.92,"lr":0.92,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v140c1200-3":
        params={"ins":"s2v140c1200-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v140c1300-1":
        params={"ins":"s2v140c1300-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v140c1300-2":
        params={"ins":"s2v140c1300-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v140c1300-3":
        params={"ins":"s2v140c1300-3","N_steps":100,"replicas":50,"T_min":0.0001,"T_max":0.62,"c_grad":0.92,"lr":0.92,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v140c1400-1":
        params={"ins":"s2v140c1400-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v140c1400-2":
        params={"ins":"s2v140c1400-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v140c1400-3":
        params={"ins":"s2v140c1400-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v140c1500-1":
        params={"ins":"s2v140c1500-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v140c1500-2":
        params={"ins":"s2v140c1500-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v140c1500-3":
        params={"ins":"s2v140c1500-3","N_steps":80,"replicas":50,"T_min":0.0001,"T_max":0.71,"c_grad":0.93,"lr":1.23,"alpha":0.31,"wd":0.014,"mom":0.83,"optimizer":"RMSprop"}
    elif instance == "s2v140c1600-1":
        params={"ins":"s2v140c1600-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v140c1600-2":
        params={"ins":"s2v140c1600-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v140c1600-3":
        params={"ins":"s2v140c1600-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v140c1700-1":
        params={"ins":"s2v140c1700-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v140c1700-2":
        params={"ins":"s2v140c1700-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v140c1700-3":
        params={"ins":"s2v140c1700-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v140c1800-1":
        params={"ins":"s2v140c1800-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v140c1800-2":
        params={"ins":"s2v140c1800-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v140c1800-3":
        params={"ins":"s2v140c1800-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v140c1900-1":
        params={"ins":"s2v140c1900-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v140c1900-2":
        params={"ins":"s2v140c1900-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v140c1900-3":
        params={"ins":"s2v140c1900-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v140c2000-1":
        params={"ins":"s2v140c2000-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v140c2000-2":
        params={"ins":"s2v140c2000-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v140c2000-3":
        params={"ins":"s2v140c2000-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v140c2100-1":
        params={"ins":"s2v140c2100-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v140c2100-2":
        params={"ins":"s2v140c2100-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v140c2100-3":
        params={"ins":"s2v140c2100-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v140c2200-1":
        params={"ins":"s2v140c2200-1","N_steps":80,"replicas":50,"T_min":0.0001,"T_max":0.71,"c_grad":0.93,"lr":1.23,"alpha":0.31,"wd":0.014,"mom":0.83,"optimizer":"RMSprop"}
    elif instance == "s2v140c2200-2":
        params={"ins":"s2v140c2200-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v140c2200-3":
        params={"ins":"s2v140c2200-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v140c2300-1":
        params={"ins":"s2v140c2300-1","N_steps":80,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":1.2,"lr":0.83,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v140c2300-2":
        params={"ins":"s2v140c2300-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v140c2300-3":
        params={"ins":"s2v140c2300-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v140c2400-1":
        params={"ins":"s2v140c2400-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v140c2400-2":
        params={"ins":"s2v140c2400-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v140c2400-3":
        params={"ins":"s2v140c2400-3","N_steps":80,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":1.2,"lr":0.83,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v140c2500-1":
        params={"ins":"s2v140c2500-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v140c2500-2":
        params={"ins":"s2v140c2500-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v140c2500-3":
        params={"ins":"s2v140c2500-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v140c2600-1":
        params={"ins":"s2v140c2600-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v140c2600-2":
        params={"ins":"s2v140c2600-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v140c2600-3":
        params={"ins":"s2v140c2600-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v160c1200-1":
        params={"ins":"s2v160c1200-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v160c1200-2":
        params={"ins":"s2v160c1200-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v160c1200-3":
        params={"ins":"s2v160c1200-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v160c1300-1":
        params={"ins":"s2v160c1300-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v160c1300-2":
        params={"ins":"s2v160c1300-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v160c1300-3":
        params={"ins":"s2v160c1300-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v160c1400-1":
        params={"ins":"s2v160c1400-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v160c1400-2":
        params={"ins":"s2v160c1400-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v160c1400-3":
        params={"ins":"s2v160c1400-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v160c1500-1":
        params={"ins":"s2v160c1500-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v160c1500-2":
        params={"ins":"s2v160c1500-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v160c1500-3":
        params={"ins":"s2v160c1500-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v160c1600-1":
        params={"ins":"s2v160c1600-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v160c1600-2":
        params={"ins":"s2v160c1600-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v160c1600-3":
        params={"ins":"s2v160c1600-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v160c1700-1":
        params={"ins":"s2v160c1700-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v160c1700-2":
        params={"ins":"s2v160c1700-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v160c1700-3":
        params={"ins":"s2v160c1700-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v160c1800-1":
        params={"ins":"s2v160c1800-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v160c1800-2":
        params={"ins":"s2v160c1800-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v160c1800-3":
        params={"ins":"s2v160c1800-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v160c1900-1":
        params={"ins":"s2v160c1900-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v160c1900-2":
        params={"ins":"s2v160c1900-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v160c1900-3":
        params={"ins":"s2v160c1900-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v160c2000-1":
        params={"ins":"s2v160c2000-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v160c2000-2":
        params={"ins":"s2v160c2000-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v160c2000-3":
        params={"ins":"s2v160c2000-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v160c2100-1":
        params={"ins":"s2v160c2100-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v160c2100-2":
        params={"ins":"s2v160c2100-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v160c2100-3":
        params={"ins":"s2v160c2100-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v160c2200-1":
        params={"ins":"s2v160c2200-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v160c2200-2":
        params={"ins":"s2v160c2200-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v160c2200-3":
        params={"ins":"s2v160c2200-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v160c2300-1":
        params={"ins":"s2v160c2300-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v160c2300-2":
        params={"ins":"s2v160c2300-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v160c2300-3":
        params={"ins":"s2v160c2300-3","N_steps":80,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":1.2,"lr":0.83,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v160c2400-1":
        params={"ins":"s2v160c2400-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v160c2400-2":
        params={"ins":"s2v160c2400-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v160c2400-3":
        params={"ins":"s2v160c2400-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v160c2500-1":
        params={"ins":"s2v160c2500-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v160c2500-2":
        params={"ins":"s2v160c2500-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v160c2500-3":
        params={"ins":"s2v160c2500-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v160c2600-1":
        params={"ins":"s2v160c2600-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v160c2600-2":
        params={"ins":"s2v160c2600-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v160c2600-3":
        params={"ins":"s2v160c2600-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v180c1200-1":
        params={"ins":"s2v180c1200-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v180c1200-2":
        params={"ins":"s2v180c1200-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v180c1200-3":
        params={"ins":"s2v180c1200-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v180c1200-4":
        params={"ins":"s2v180c1200-4","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v180c1300-1":
        params={"ins":"s2v180c1300-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v180c1300-2":
        params={"ins":"s2v180c1300-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v180c1300-3":
        params={"ins":"s2v180c1300-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v180c1300-4":
        params={"ins":"s2v180c1300-4","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v180c1400-1":
        params={"ins":"s2v180c1400-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v180c1400-2":
        params={"ins":"s2v180c1400-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v180c1400-3":
        params={"ins":"s2v180c1400-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v180c1400-4":
        params={"ins":"s2v180c1400-4","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v180c1500-1":
        params={"ins":"s2v180c1500-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v180c1500-2":
        params={"ins":"s2v180c1500-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v180c1500-3":
        params={"ins":"s2v180c1500-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v180c1500-4":
        params={"ins":"s2v180c1500-4","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v180c1600-1":
        params={"ins":"s2v180c1600-1","N_steps":80,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":1.2,"lr":0.83,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v180c1600-2":
        params={"ins":"s2v180c1600-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v180c1600-3":
        params={"ins":"s2v180c1600-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v180c1600-4":
        params={"ins":"s2v180c1600-4","N_steps":80,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":1.2,"lr":0.83,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v180c1700-1":
        params={"ins":"s2v180c1700-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v180c1700-2":
        params={"ins":"s2v180c1700-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v180c1700-3":
        params={"ins":"s2v180c1700-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v180c1700-4":
        params={"ins":"s2v180c1700-4","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v180c1800-1":
        params={"ins":"s2v180c1800-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v180c1800-2":
        params={"ins":"s2v180c1800-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v180c1800-3":
        params={"ins":"s2v180c1800-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v180c1800-4":
        params={"ins":"s2v180c1800-4","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v180c1900-1":
        params={"ins":"s2v180c1900-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v180c1900-2":
        params={"ins":"s2v180c1900-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v180c1900-3":
        params={"ins":"s2v180c1900-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v180c1900-4":
        params={"ins":"s2v180c1900-4","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v180c2000-1":
        params={"ins":"s2v180c2000-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v180c2000-2":
        params={"ins":"s2v180c2000-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v180c2000-3":
        params={"ins":"s2v180c2000-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v180c2000-4":
        params={"ins":"s2v180c2000-4","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v180c2100-1":
        params={"ins":"s2v180c2100-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v180c2100-2":
        params={"ins":"s2v180c2100-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v180c2100-3":
        params={"ins":"s2v180c2100-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v180c2100-4":
        params={"ins":"s2v180c2100-4","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v180c2200-1":
        params={"ins":"s2v180c2200-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v180c2200-2":
        params={"ins":"s2v180c2200-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v180c2200-3":
        params={"ins":"s2v180c2200-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v180c2200-4":
        params={"ins":"s2v180c2200-4","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v200c1200-1":
        params={"ins":"s2v200c1200-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v200c1200-2":
        params={"ins":"s2v200c1200-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v200c1200-3":
        params={"ins":"s2v200c1200-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v200c1200-4":
        params={"ins":"s2v200c1200-4","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v200c1200-5":
        params={"ins":"s2v200c1200-5","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v200c1200-6":
        params={"ins":"s2v200c1200-6","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v200c1200-7":
        params={"ins":"s2v200c1200-7","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v200c1300-1":
        params={"ins":"s2v200c1300-1","N_steps":80,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":1.2,"lr":0.83,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v200c1300-2":
        params={"ins":"s2v200c1300-2","N_steps":80,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":1.2,"lr":0.83,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v200c1300-3":
        params={"ins":"s2v200c1300-3","N_steps":80,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":1.2,"lr":0.83,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v200c1300-4":
        params={"ins":"s2v200c1300-4","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v200c1300-5":
        params={"ins":"s2v200c1300-5","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v200c1300-6":
        params={"ins":"s2v200c1300-6","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v200c1300-7":
        params={"ins":"s2v200c1300-7","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v200c1400-1":
        params={"ins":"s2v200c1400-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v200c1400-2":
        params={"ins":"s2v200c1400-2","N_steps":20,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v200c1400-3":
        params={"ins":"s2v200c1400-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v200c1400-4":
        params={"ins":"s2v200c1400-4","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v200c1400-5":
        params={"ins":"s2v200c1400-5","N_steps":80,"replicas":50,"T_min":0.0001,"T_max":0.83,"c_grad":1.0,"lr":0.73,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v200c1400-6":
        params={"ins":"s2v200c1400-6","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v200c1400-7":
        params={"ins":"s2v200c1400-7","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v200c1500-1":
        params={"ins":"s2v200c1500-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v200c1500-2":
        params={"ins":"s2v200c1500-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v200c1500-3":
        params={"ins":"s2v200c1500-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v200c1500-4":
        params={"ins":"s2v200c1500-4","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v200c1500-5":
        params={"ins":"s2v200c1500-5","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v200c1500-6":
        params={"ins":"s2v200c1500-6","N_steps":80,"replicas":50,"T_min":0.0001,"T_max":0.83,"c_grad":1.0,"lr":0.73,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v200c1500-7":
        params={"ins":"s2v200c1500-7","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v200c1600-1":
        params={"ins":"s2v200c1600-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v200c1600-2":
        params={"ins":"s2v200c1600-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v200c1600-3":
        params={"ins":"s2v200c1600-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v200c1600-4":
        params={"ins":"s2v200c1600-4","N_steps":80,"replicas":50,"T_min":0.0001,"T_max":0.83,"c_grad":1.0,"lr":0.73,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v200c1600-5":
        params={"ins":"s2v200c1600-5","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v200c1600-6":
        params={"ins":"s2v200c1600-6","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v200c1600-7":
        params={"ins":"s2v200c1600-7","N_steps":80,"replicas":50,"T_min":0.0001,"T_max":0.83,"c_grad":1.0,"lr":0.73,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v200c1700-1":
        params={"ins":"s2v200c1700-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v200c1700-2":
        params={"ins":"s2v200c1700-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v200c1700-3":
        params={"ins":"s2v200c1700-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v200c1700-4":
        params={"ins":"s2v200c1700-4","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v200c1700-5":
        params={"ins":"s2v200c1700-5","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v200c1700-6":
        params={"ins":"s2v200c1700-6","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v200c1700-7":
        params={"ins":"s2v200c1700-7","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v200c1800-1":
        params={"ins":"s2v200c1800-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v200c1800-2":
        params={"ins":"s2v200c1800-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v200c1800-3":
        params={"ins":"s2v200c1800-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v200c1800-4":
        params={"ins":"s2v200c1800-4","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v200c1800-5":
        params={"ins":"s2v200c1800-5","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v200c1800-6":
        params={"ins":"s2v200c1800-6","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s2v200c1800-7":
        params={"ins":"s2v200c1800-7","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v110c1000-1":
        params={"ins":"s3v110c1000-1","N_steps":80,"replicas":50,"T_min":0.0001,"T_max":0.83,"c_grad":1.0,"lr":0.73,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v110c1000-10":
        params={"ins":"s3v110c1000-10","N_steps":80,"replicas":50,"T_min":0.0001,"T_max":0.83,"c_grad":1.0,"lr":0.73,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v110c1000-2":
        params={"ins":"s3v110c1000-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v110c1000-3":
        params={"ins":"s3v110c1000-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v110c1000-4":
        params={"ins":"s3v110c1000-4","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v110c1000-5":
        params={"ins":"s3v110c1000-5","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v110c1000-6":
        params={"ins":"s3v110c1000-6","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v110c1000-7":
        params={"ins":"s3v110c1000-7","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v110c1000-8":
        params={"ins":"s3v110c1000-8","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v110c1000-9":
        params={"ins":"s3v110c1000-9","N_steps":80,"replicas":50,"T_min":0.0001,"T_max":0.83,"c_grad":1.0,"lr":0.73,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v110c1100-1":
        params={"ins":"s3v110c1100-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v110c1100-10":
        params={"ins":"s3v110c1100-10","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v110c1100-2":
        params={"ins":"s3v110c1100-2","N_steps":80,"replicas":50,"T_min":0.0001,"T_max":0.83,"c_grad":1.0,"lr":0.73,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v110c1100-3":
        params={"ins":"s3v110c1100-3","N_steps":80,"replicas":50,"T_min":0.0001,"T_max":0.83,"c_grad":1.0,"lr":0.73,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v110c1100-4":
        params={"ins":"s3v110c1100-4","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v110c1100-5":
        params={"ins":"s3v110c1100-5","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v110c1100-6":
        params={"ins":"s3v110c1100-6","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v110c1100-7":
        params={"ins":"s3v110c1100-7","N_steps":80,"replicas":50,"T_min":0.0001,"T_max":0.83,"c_grad":1.0,"lr":0.73,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v110c1100-8":
        params={"ins":"s3v110c1100-8","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v110c1100-9":
        params={"ins":"s3v110c1100-9","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v110c700-1":
        params={"ins":"s3v110c700-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v110c700-10":
        params={"ins":"s3v110c700-10","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v110c700-2":
        params={"ins":"s3v110c700-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v110c700-3":
        params={"ins":"s3v110c700-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v110c700-4":
        params={"ins":"s3v110c700-4","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v110c700-5":
        params={"ins":"s3v110c700-5","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v110c700-6":
        params={"ins":"s3v110c700-6","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v110c700-7":
        params={"ins":"s3v110c700-7","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v110c700-8":
        params={"ins":"s3v110c700-8","N_steps":80,"replicas":50,"T_min":0.0001,"T_max":0.83,"c_grad":1.0,"lr":0.73,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v110c700-9":
        params={"ins":"s3v110c700-9","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v110c800-1":
        params={"ins":"s3v110c800-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v110c800-10":
        params={"ins":"s3v110c800-10","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v110c800-2":
        params={"ins":"s3v110c800-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v110c800-3":
        params={"ins":"s3v110c800-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v110c800-4":
        params={"ins":"s3v110c800-4","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v110c800-5":
        params={"ins":"s3v110c800-5","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v110c800-6":
        params={"ins":"s3v110c800-6","N_steps":80,"replicas":50,"T_min":0.0001,"T_max":0.83,"c_grad":1.0,"lr":0.73,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v110c800-7":
        params={"ins":"s3v110c800-7","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v110c800-8":
        params={"ins":"s3v110c800-8","N_steps":80,"replicas":50,"T_min":0.0001,"T_max":0.83,"c_grad":1.0,"lr":0.73,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v110c800-9":
        params={"ins":"s3v110c800-9","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v110c900-1":
        params={"ins":"s3v110c900-1","N_steps":80,"replicas":50,"T_min":0.0001,"T_max":0.83,"c_grad":1.0,"lr":0.73,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v110c900-10":
        params={"ins":"s3v110c900-10","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v110c900-2":
        params={"ins":"s3v110c900-2","N_steps":80,"replicas":50,"T_min":0.0001,"T_max":0.63,"c_grad":3.13,"lr":2.2,"alpha":0.16,"wd":0.013,"mom":0.56,"optimizer":"RMSprop"}
    elif instance == "s3v110c900-3":
        params={"ins":"s3v110c900-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v110c900-4":
        params={"ins":"s3v110c900-4","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v110c900-5":
        params={"ins":"s3v110c900-5","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v110c900-6":
        params={"ins":"s3v110c900-6","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v110c900-7":
        params={"ins":"s3v110c900-7","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v110c900-8":
        params={"ins":"s3v110c900-8","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v110c900-9":
        params={"ins":"s3v110c900-9","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v70c1000-1":
        params={"ins":"s3v70c1000-1","N_steps":80,"replicas":50,"T_min":0.0001,"T_max":0.63,"c_grad":3.13,"lr":2.2,"alpha":0.16,"wd":0.013,"mom":0.56,"optimizer":"RMSprop"}
    elif instance == "s3v70c1000-2":
        params={"ins":"s3v70c1000-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v70c1000-3":
        params={"ins":"s3v70c1000-3","N_steps":80,"replicas":50,"T_min":0.0001,"T_max":0.63,"c_grad":3.13,"lr":2.2,"alpha":0.16,"wd":0.013,"mom":0.56,"optimizer":"RMSprop"}
    elif instance == "s3v70c1000-4":
        params={"ins":"s3v70c1000-4","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v70c1000-5":
        params={"ins":"s3v70c1000-5","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v70c1100-1":
        params={"ins":"s3v70c1100-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v70c1100-2":
        params={"ins":"s3v70c1100-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v70c1100-3":
        params={"ins":"s3v70c1100-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v70c1100-4":
        params={"ins":"s3v70c1100-4","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v70c1100-5":
        params={"ins":"s3v70c1100-5","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v70c1200-1":
        params={"ins":"s3v70c1200-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v70c1200-2":
        params={"ins":"s3v70c1200-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v70c1200-3":
        params={"ins":"s3v70c1200-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v70c1200-4":
        params={"ins":"s3v70c1200-4","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v70c1200-5":
        params={"ins":"s3v70c1200-5","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v70c1300-1":
        params={"ins":"s3v70c1300-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v70c1300-2":
        params={"ins":"s3v70c1300-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v70c1300-3":
        params={"ins":"s3v70c1300-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v70c1300-4":
        params={"ins":"s3v70c1300-4","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v70c1300-5":
        params={"ins":"s3v70c1300-5","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v70c1400-1":
        params={"ins":"s3v70c1400-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v70c1400-2":
        params={"ins":"s3v70c1400-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v70c1400-3":
        params={"ins":"s3v70c1400-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v70c1400-4":
        params={"ins":"s3v70c1400-4","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v70c1400-5":
        params={"ins":"s3v70c1400-5","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v70c1500-1":
        params={"ins":"s3v70c1500-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v70c1500-2":
        params={"ins":"s3v70c1500-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v70c1500-3":
        params={"ins":"s3v70c1500-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v70c1500-4":
        params={"ins":"s3v70c1500-4","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v70c1500-5":
        params={"ins":"s3v70c1500-5","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v70c700-1":
        params={"ins":"s3v70c700-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v70c700-2":
        params={"ins":"s3v70c700-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v70c700-3":
        params={"ins":"s3v70c700-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v70c700-4":
        params={"ins":"s3v70c700-4","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v70c700-5":
        params={"ins":"s3v70c700-5","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v70c800-1":
        params={"ins":"s3v70c800-1","N_steps":80,"replicas":50,"T_min":0.0001,"T_max":0.63,"c_grad":3.13,"lr":2.2,"alpha":0.16,"wd":0.013,"mom":0.56,"optimizer":"RMSprop"}
    elif instance == "s3v70c800-2":
        params={"ins":"s3v70c800-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v70c800-3":
        params={"ins":"s3v70c800-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v70c800-4":
        params={"ins":"s3v70c800-4","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v70c800-5":
        params={"ins":"s3v70c800-5","N_steps":80,"replicas":50,"T_min":0.0001,"T_max":0.63,"c_grad":3.13,"lr":2.2,"alpha":0.16,"wd":0.013,"mom":0.56,"optimizer":"RMSprop"}
    elif instance == "s3v70c900-1":
        params={"ins":"s3v70c900-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v70c900-2":
        params={"ins":"s3v70c900-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v70c900-3":
        params={"ins":"s3v70c900-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v70c900-4":
        params={"ins":"s3v70c900-4","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v70c900-5":
        params={"ins":"s3v70c900-5","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v90c1000-1":
        params={"ins":"s3v90c1000-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v90c1000-2":
        params={"ins":"s3v90c1000-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v90c1000-3":
        params={"ins":"s3v90c1000-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v90c1000-4":
        params={"ins":"s3v90c1000-4","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v90c1000-5":
        params={"ins":"s3v90c1000-5","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v90c1000-6":
        params={"ins":"s3v90c1000-6","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v90c1000-7":
        params={"ins":"s3v90c1000-7","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v90c1100-1":
        params={"ins":"s3v90c1100-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v90c1100-2":
        params={"ins":"s3v90c1100-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v90c1100-3":
        params={"ins":"s3v90c1100-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v90c1100-4":
        params={"ins":"s3v90c1100-4","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v90c1100-5":
        params={"ins":"s3v90c1100-5","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v90c1100-6":
        params={"ins":"s3v90c1100-6","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v90c1100-7":
        params={"ins":"s3v90c1100-7","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v90c1200-1":
        params={"ins":"s3v90c1200-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v90c1200-2":
        params={"ins":"s3v90c1200-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v90c1200-3":
        params={"ins":"s3v90c1200-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v90c1200-4":
        params={"ins":"s3v90c1200-4","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v90c1200-5":
        params={"ins":"s3v90c1200-5","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v90c1200-6":
        params={"ins":"s3v90c1200-6","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v90c1200-7":
        params={"ins":"s3v90c1200-7","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v90c1300-1":
        params={"ins":"s3v90c1300-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v90c1300-2":
        params={"ins":"s3v90c1300-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v90c1300-3":
        params={"ins":"s3v90c1300-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v90c1300-4":
        params={"ins":"s3v90c1300-4","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v90c1300-5":
        params={"ins":"s3v90c1300-5","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v90c1300-6":
        params={"ins":"s3v90c1300-6","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v90c1300-7":
        params={"ins":"s3v90c1300-7","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v90c700-1":
        params={"ins":"s3v90c700-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v90c700-2":
        params={"ins":"s3v90c700-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v90c700-3":
        params={"ins":"s3v90c700-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v90c700-4":
        params={"ins":"s3v90c700-4","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v90c700-5":
        params={"ins":"s3v90c700-5","N_steps":80,"replicas":50,"T_min":0.0001,"T_max":0.63,"c_grad":3.13,"lr":2.2,"alpha":0.16,"wd":0.013,"mom":0.56,"optimizer":"RMSprop"}
    elif instance == "s3v90c700-6":
        params={"ins":"s3v90c700-6","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v90c700-7":
        params={"ins":"s3v90c700-7","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v90c800-1":
        params={"ins":"s3v90c800-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v90c800-2":
        params={"ins":"s3v90c800-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v90c800-3":
        params={"ins":"s3v90c800-3","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v90c800-4":
        params={"ins":"s3v90c800-4","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v90c800-5":
        params={"ins":"s3v90c800-5","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v90c800-6":
        params={"ins":"s3v90c800-6","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v90c800-7":
        params={"ins":"s3v90c800-7","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v90c900-1":
        params={"ins":"s3v90c900-1","N_steps":80,"replicas":50,"T_min":0.0001,"T_max":0.63,"c_grad":3.13,"lr":2.2,"alpha":0.16,"wd":0.013,"mom":0.56,"optimizer":"RMSprop"}
    elif instance == "s3v90c900-2":
        params={"ins":"s3v90c900-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v90c900-3":
        params={"ins":"s3v90c900-3","N_steps":80,"replicas":50,"T_min":0.0001,"T_max":0.63,"c_grad":3.13,"lr":2.2,"alpha":0.16,"wd":0.013,"mom":0.56,"optimizer":"RMSprop"}
    elif instance == "s3v90c900-4":
        params={"ins":"s3v90c900-4","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v90c900-5":
        params={"ins":"s3v90c900-5","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v90c900-6":
        params={"ins":"s3v90c900-6","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "s3v90c900-7":
        params={"ins":"s3v90c900-7","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":0.8,"lr":0.8,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "HG-3SAT-V250-C1000-1":
        params={"ins":"HG-3SAT-V250-C1000-1","N_steps":80,"replicas":50,"T_min":0.0001,"T_max":0.66,"c_grad":0.7,"lr":1.06,"alpha":0.26,"wd":0.035,"mom":0.56,"optimizer":"RMSprop"}
    elif instance == "HG-3SAT-V250-C1000-10":
        params={"ins":"HG-3SAT-V250-C1000-10","N_steps":80,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":1.01,"lr":0.73,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "HG-3SAT-V250-C1000-100":
        params={"ins":"HG-3SAT-V250-C1000-100","N_steps":80,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":1.01,"lr":0.73,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "HG-3SAT-V250-C1000-11":
        params={"ins":"HG-3SAT-V250-C1000-11","N_steps":80,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":1.01,"lr":0.73,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "HG-3SAT-V250-C1000-12":
        params={"ins":"HG-3SAT-V250-C1000-12","N_steps":80,"replicas":50,"T_min":0.0001,"T_max":0.6,"c_grad":1.01,"lr":0.73,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "HG-3SAT-V250-C1000-13":
        params={"ins":"HG-3SAT-V250-C1000-13","N_steps":100,"replicas":50,"T_min":0.0001,"T_max":0.83,"c_grad":0.92,"lr":0.64,"alpha":0.3,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "HG-3SAT-V250-C1000-14":
        params={"ins":"HG-3SAT-V250-C1000-14","N_steps":500,"replicas":800,"T_min":0.0001,"T_max":0.9,"c_grad":0.6,"lr":0.7,"alpha":0.1,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "HG-3SAT-V250-C1000-15":
        params={"ins":"HG-3SAT-V250-C1000-15","N_steps":90,"replicas":50,"T_min":0.0001,"T_max":0.64,"c_grad":0.72,"lr":0.64,"alpha":0.12,"wd":0.02,"mom":0.876,"optimizer":"RMSprop"}
    elif instance == "HG-3SAT-V250-C1000-16":
        params={"ins":"HG-3SAT-V250-C1000-16","N_steps":130,"replicas":60,"T_min":0.0001,"T_max":0.8,"c_grad":0.6,"lr":0.56,"alpha":0.1,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "HG-3SAT-V250-C1000-17":
        params={"ins":"HG-3SAT-V250-C1000-17","N_steps":130,"replicas":80,"T_min":0.0001,"T_max":0.8,"c_grad":0.6,"lr":0.56,"alpha":0.1,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "HG-3SAT-V250-C1000-18":
        params={"ins":"HG-3SAT-V250-C1000-18","N_steps":200,"replicas":100,"T_min":0.0001,"T_max":0.7,"c_grad":1.28,"lr":0.75,"alpha":0.07,"wd":0.02,"mom":0.83,"optimizer":"RMSprop"}
    elif instance == "HG-3SAT-V250-C1000-19":
        params={"ins":"HG-3SAT-V250-C1000-19","N_steps":300,"replicas":100,"T_min":0.0001,"T_max":0.76,"c_grad":0.85,"lr":0.56,"alpha":0.1,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "HG-3SAT-V250-C1000-2":
        params={"ins":"HG-3SAT-V250-C1000-2","N_steps":300,"replicas":100,"T_min":0.0001,"T_max":0.8,"c_grad":0.53,"lr":0.56,"alpha":0.1,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "HG-3SAT-V250-C1000-20":
        params={"ins":"HG-3SAT-V250-C1000-20","N_steps":300,"replicas":100,"T_min":0.0001,"T_max":0.83,"c_grad":0.67,"lr":0.56,"alpha":0.1,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "HG-3SAT-V250-C1000-21":
        params={"ins":"HG-3SAT-V250-C1000-21","N_steps":300,"replicas":100,"T_min":0.0001,"T_max":0.83,"c_grad":0.81,"lr":0.75,"alpha":0.1,"wd":0.02,"mom":0.83,"optimizer":"RMSprop"}
    elif instance == "HG-3SAT-V250-C1000-22":
        params={"ins":"HG-3SAT-V250-C1000-22","N_steps":300,"replicas":200,"T_min":0.0001,"T_max":0.8,"c_grad":0.85,"lr":0.56,"alpha":0.1,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "HG-3SAT-V250-C1000-23":
        params={"ins":"HG-3SAT-V250-C1000-23","N_steps":300,"replicas":200,"T_min":0.0001,"T_max":0.8,"c_grad":0.85,"lr":0.56,"alpha":0.1,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "HG-3SAT-V250-C1000-24":
        params={"ins":"HG-3SAT-V250-C1000-24","N_steps":300,"replicas":200,"T_min":0.0001,"T_max":0.8,"c_grad":0.6,"lr":0.72,"alpha":0.1,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "HG-3SAT-V250-C1000-3":
        params={"ins":"HG-3SAT-V250-C1000-3","N_steps":300,"replicas":200,"T_min":0.0001,"T_max":0.8,"c_grad":0.6,"lr":0.72,"alpha":0.1,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "HG-3SAT-V250-C1000-4":
        params={"ins":"HG-3SAT-V250-C1000-4","N_steps":300,"replicas":200,"T_min":0.0001,"T_max":0.64,"c_grad":0.53,"lr":0.62,"alpha":0.12,"wd":0.016,"mom":0.92,"optimizer":"RMSprop"}
    elif instance == "HG-3SAT-V250-C1000-5":
        params={"ins":"HG-3SAT-V250-C1000-5","N_steps":300,"replicas":200,"T_min":0.0001,"T_max":0.64,"c_grad":0.81,"lr":0.56,"alpha":0.09,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "HG-3SAT-V250-C1000-6":
        params={"ins":"HG-3SAT-V250-C1000-6","N_steps":300,"replicas":200,"T_min":0.0001,"T_max":0.64,"c_grad":0.81,"lr":0.56,"alpha":0.09,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "HG-3SAT-V250-C1000-7":
        params={"ins":"HG-3SAT-V250-C1000-7","N_steps":300,"replicas":200,"T_min":0.0001,"T_max":0.64,"c_grad":0.81,"lr":0.56,"alpha":0.09,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "HG-3SAT-V250-C1000-8":
        params={"ins":"HG-3SAT-V250-C1000-8","N_steps":300,"replicas":200,"T_min":0.0001,"T_max":0.64,"c_grad":0.81,"lr":0.56,"alpha":0.09,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "HG-3SAT-V250-C1000-9":
        params={"ins":"HG-3SAT-V250-C1000-9","N_steps":300,"replicas":200,"T_min":0.0001,"T_max":0.64,"c_grad":0.81,"lr":0.56,"alpha":0.09,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "HG-3SAT-V300-C1200-1":
        params={"ins":"HG-3SAT-V300-C1200-1","N_steps":100,"replicas":200,"T_min":0.0001,"T_max":0.91,"c_grad":0.76,"lr":0.64,"alpha":0.08,"wd":0.02,"mom":0.63,"optimizer":"RMSprop"}
    elif instance == "HG-3SAT-V300-C1200-10":
        params={"ins":"HG-3SAT-V300-C1200-10","N_steps":200,"replicas":200,"T_min":0.0001,"T_max":0.67,"c_grad":0.55,"lr":0.39,"alpha":0.08,"wd":0.02,"mom":0.91,"optimizer":"RMSprop"}
    elif instance == "HG-3SAT-V300-C1200-100":
        params={"ins":"HG-3SAT-V300-C1200-100","N_steps":100,"replicas":200,"T_min":0.0001,"T_max":0.67,"c_grad":0.55,"lr":0.39,"alpha":0.08,"wd":0.02,"mom":0.91,"optimizer":"RMSprop"}
    elif instance == "HG-3SAT-V300-C1200-11":
        params={"ins":"HG-3SAT-V300-C1200-11","N_steps":200,"replicas":200,"T_min":0.0001,"T_max":0.7,"c_grad":0.8,"lr":0.73,"alpha":0.11,"wd":0.02,"mom":0.63,"optimizer":"RMSprop"}
    elif instance == "HG-3SAT-V300-C1200-12":
        params={"ins":"HG-3SAT-V300-C1200-12","N_steps":100,"replicas":200,"T_min":0.0001,"T_max":0.82,"c_grad":0.56,"lr":0.77,"alpha":0.09,"wd":0.02,"mom":0.7,"optimizer":"RMSprop"}
    elif instance == "HG-3SAT-V300-C1200-13":
        params={"ins":"HG-3SAT-V300-C1200-13","N_steps":200,"replicas":200,"T_min":0.0001,"T_max":0.54,"c_grad":1.29,"lr":1.17,"alpha":0.13,"wd":0.02,"mom":0.72,"optimizer":"RMSprop"}
    elif instance == "HG-3SAT-V300-C1200-14":
        params={"ins":"HG-3SAT-V300-C1200-14","N_steps":100,"replicas":100,"T_min":0.0001,"T_max":0.7,"c_grad":0.85,"lr":0.56,"alpha":0.1,"wd":0.03,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "HG-3SAT-V300-C1200-15":
        params={"ins":"HG-3SAT-V300-C1200-15","N_steps":100,"replicas":100,"T_min":0.0001,"T_max":0.7,"c_grad":0.96,"lr":0.54,"alpha":0.12,"wd":0.02,"mom":0.88,"optimizer":"RMSprop"}
    elif instance == "HG-3SAT-V300-C1200-16":
        params={"ins":"HG-3SAT-V300-C1200-16","N_steps":100,"replicas":100,"T_min":0.0001,"T_max":0.8,"c_grad":0.76,"lr":0.56,"alpha":0.1,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "HG-3SAT-V300-C1200-17":
        params={"ins":"HG-3SAT-V300-C1200-17","N_steps":100,"replicas":100,"T_min":0.0001,"T_max":0.64,"c_grad":0.57,"lr":0.56,"alpha":0.1,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "HG-3SAT-V300-C1200-18":
        params={"ins":"HG-3SAT-V300-C1200-18","N_steps":100,"replicas":100,"T_min":0.0001,"T_max":0.8,"c_grad":0.71,"lr":0.56,"alpha":0.1,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "HG-3SAT-V300-C1200-19":
        params={"ins":"HG-3SAT-V300-C1200-19","N_steps":100,"replicas":100,"T_min":0.0001,"T_max":0.53,"c_grad":0.55,"lr":0.55,"alpha":0.09,"wd":0.02,"mom":0.88,"optimizer":"RMSprop"}
    elif instance == "HG-3SAT-V300-C1200-2":
        params={"ins":"HG-3SAT-V300-C1200-2","N_steps":500,"replicas":200,"T_min":0.0001,"T_max":0.45,"c_grad":0.94,"lr":0.68,"alpha":0.08,"wd":0.03,"mom":0.92,"optimizer":"RMSprop"}
    elif instance == "HG-3SAT-V300-C1200-20":
        params={"ins":"HG-3SAT-V300-C1200-20","N_steps":200,"replicas":200,"T_min":0.0001,"T_max":0.8,"c_grad":0.6,"lr":0.56,"alpha":0.1,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "HG-3SAT-V300-C1200-21":
        params={"ins":"HG-3SAT-V300-C1200-21","N_steps":200,"replicas":200,"T_min":0.0001,"T_max":0.73,"c_grad":0.51,"lr":0.54,"alpha":0.13,"wd":0.02,"mom":0.88,"optimizer":"RMSprop"}
    elif instance == "HG-3SAT-V300-C1200-22":
        params={"ins":"HG-3SAT-V300-C1200-22","N_steps":200,"replicas":200,"T_min":0.0001,"T_max":0.7,"c_grad":0.48,"lr":0.56,"alpha":0.1,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "HG-3SAT-V300-C1200-23":
        params={"ins":"HG-3SAT-V300-C1200-23","N_steps":100,"replicas":200,"T_min":0.0001,"T_max":0.56,"c_grad":0.42,"lr":0.84,"alpha":0.07,"wd":0.02,"mom":0.75,"optimizer":"RMSprop"}
    elif instance == "HG-3SAT-V300-C1200-24":
        params={"ins":"HG-3SAT-V300-C1200-24","N_steps":100,"replicas":200,"T_min":0.0001,"T_max":0.8,"c_grad":0.62,"lr":0.56,"alpha":0.1,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "HG-3SAT-V300-C1200-3":
        params={"ins":"HG-3SAT-V300-C1200-3","N_steps":100,"replicas":200,"T_min":0.0001,"T_max":0.92,"c_grad":0.59,"lr":0.85,"alpha":0.13,"wd":0.02,"mom":0.54,"optimizer":"RMSprop"}
    elif instance == "HG-3SAT-V300-C1200-4":
        params={"ins":"HG-3SAT-V300-C1200-4","N_steps":100,"replicas":200,"T_min":0.0001,"T_max":0.8,"c_grad":0.85,"lr":0.56,"alpha":0.1,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "HG-3SAT-V300-C1200-5":
        params={"ins":"HG-3SAT-V300-C1200-5","N_steps":100,"replicas":200,"T_min":0.0001,"T_max":0.8,"c_grad":0.71,"lr":0.56,"alpha":0.1,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "HG-3SAT-V300-C1200-6":
        params={"ins":"HG-3SAT-V300-C1200-6","N_steps":100,"replicas":200,"T_min":0.0001,"T_max":0.8,"c_grad":0.6,"lr":0.56,"alpha":0.1,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "HG-3SAT-V300-C1200-7":
        params={"ins":"HG-3SAT-V300-C1200-7","N_steps":100,"replicas":200,"T_min":0.0001,"T_max":0.64,"c_grad":0.48,"lr":0.54,"alpha":0.09,"wd":0.02,"mom":0.88,"optimizer":"RMSprop"}
    elif instance == "HG-3SAT-V300-C1200-8":
        params={"ins":"HG-3SAT-V300-C1200-8","N_steps":100,"replicas":200,"T_min":0.0001,"T_max":0.76,"c_grad":0.53,"lr":0.84,"alpha":0.1,"wd":0.02,"mom":0.63,"optimizer":"RMSprop"}
    elif instance == "HG-3SAT-V300-C1200-9":
        params={"ins":"HG-3SAT-V300-C1200-9","N_steps":100,"replicas":200,"T_min":0.0001,"T_max":0.64,"c_grad":0.79,"lr":0.49,"alpha":0.12,"wd":0.02,"mom":0.92,"optimizer":"RMSprop"}
    elif instance == "HG-4SAT-V100-C900-14":
        params={"ins":"HG-4SAT-V100-C900-14","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.7,"c_grad":0.85,"lr":0.63,"alpha":0.08,"wd":0.02,"mom":0.67,"optimizer":"RMSprop"}
    elif instance == "HG-4SAT-V100-C900-19":
        params={"ins":"HG-4SAT-V100-C900-19","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.8,"c_grad":1.03,"lr":0.62,"alpha":0.09,"wd":0.02,"mom":0.77,"optimizer":"RMSprop"}
    elif instance == "HG-4SAT-V100-C900-2":
        params={"ins":"HG-4SAT-V100-C900-2","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.85,"c_grad":0.66,"lr":0.65,"alpha":0.06,"wd":0.03,"mom":0.44,"optimizer":"RMSprop"}
    elif instance == "HG-4SAT-V100-C900-20":
        params={"ins":"HG-4SAT-V100-C900-20","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.66,"c_grad":1.03,"lr":0.49,"alpha":0.05,"wd":0.03,"mom":0.5,"optimizer":"RMSprop"}
    elif instance == "HG-4SAT-V100-C900-23":
        params={"ins":"HG-4SAT-V100-C900-23","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":1.31,"c_grad":1.26,"lr":0.71,"alpha":0.09,"wd":0.03,"mom":0.41,"optimizer":"RMSprop"}
    elif instance == "HG-4SAT-V100-C900-4":
        params={"ins":"HG-4SAT-V100-C900-4","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.64,"c_grad":0.48,"lr":0.56,"alpha":0.1,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "HG-4SAT-V100-C900-7":
        params={"ins":"HG-4SAT-V100-C900-7","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.8,"c_grad":0.6,"lr":0.56,"alpha":0.1,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "HG-4SAT-V150-C1350-1":
        params={"ins":"HG-4SAT-V150-C1350-1","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.83,"c_grad":0.62,"lr":0.56,"alpha":0.1,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "HG-4SAT-V150-C1350-10":
        params={"ins":"HG-4SAT-V150-C1350-10","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.76,"c_grad":0.75,"lr":0.77,"alpha":0.07,"wd":0.03,"mom":0.75,"optimizer":"RMSprop"}
    elif instance == "HG-4SAT-V150-C1350-100":
        params={"ins":"HG-4SAT-V150-C1350-100","N_steps":70,"replicas":50,"T_min":0.0001,"T_max":0.76,"c_grad":0.85,"lr":0.61,"alpha":0.11,"wd":0.02,"mom":0.75,"optimizer":"RMSprop"}
    elif instance == "HG-4SAT-V150-C1350-11":
        params={"ins":"HG-4SAT-V150-C1350-11","N_steps":100,"replicas":100,"T_min":0.0001,"T_max":0.76,"c_grad":0.9,"lr":0.54,"alpha":0.12,"wd":0.03,"mom":0.88,"optimizer":"RMSprop"}
    elif instance == "HG-4SAT-V150-C1350-12":
        params={"ins":"HG-4SAT-V150-C1350-12","N_steps":100,"replicas":100,"T_min":0.0001,"T_max":0.64,"c_grad":0.62,"lr":0.67,"alpha":0.1,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "HG-4SAT-V150-C1350-13":
        params={"ins":"HG-4SAT-V150-C1350-13","N_steps":80,"replicas":100,"T_min":0.0001,"T_max":0.7,"c_grad":0.76,"lr":0.56,"alpha":0.1,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "HG-4SAT-V150-C1350-14":
        params={"ins":"HG-4SAT-V150-C1350-14","N_steps":80,"replicas":100,"T_min":0.0001,"T_max":0.8,"c_grad":0.6,"lr":0.72,"alpha":0.1,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "HG-4SAT-V150-C1350-15":
        params={"ins":"HG-4SAT-V150-C1350-15","N_steps":80,"replicas":100,"T_min":0.0001,"T_max":0.83,"c_grad":0.96,"lr":0.49,"alpha":0.12,"wd":0.02,"mom":0.88,"optimizer":"RMSprop"}
    elif instance == "HG-4SAT-V150-C1350-16":
        params={"ins":"HG-4SAT-V150-C1350-16","N_steps":100,"replicas":100,"T_min":0.0001,"T_max":0.8,"c_grad":0.76,"lr":0.56,"alpha":0.1,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "HG-4SAT-V150-C1350-17":
        params={"ins":"HG-4SAT-V150-C1350-17","N_steps":80,"replicas":100,"T_min":0.0001,"T_max":0.7,"c_grad":0.57,"lr":0.71,"alpha":0.1,"wd":0.02,"mom":0.71,"optimizer":"RMSprop"}
    elif instance == "HG-4SAT-V150-C1350-18":
        params={"ins":"HG-4SAT-V150-C1350-18","N_steps":100,"replicas":100,"T_min":0.0001,"T_max":0.64,"c_grad":0.81,"lr":0.56,"alpha":0.09,"wd":0.03,"mom":0.79,"optimizer":"RMSprop"}
    elif instance == "HG-4SAT-V150-C1350-19":
        params={"ins":"HG-4SAT-V150-C1350-19","N_steps":100,"replicas":100,"T_min":0.0001,"T_max":0.8,"c_grad":0.6,"lr":0.61,"alpha":0.1,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "HG-4SAT-V150-C1350-2":
        params={"ins":"HG-4SAT-V150-C1350-2","N_steps":100,"replicas":100,"T_min":0.0001,"T_max":0.64,"c_grad":0.57,"lr":0.61,"alpha":0.09,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "HG-4SAT-V150-C1350-20":
        params={"ins":"HG-4SAT-V150-C1350-20","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.83,"c_grad":0.64,"lr":0.49,"alpha":0.09,"wd":0.02,"mom":0.67,"optimizer":"RMSprop"}
    elif instance == "HG-4SAT-V150-C1350-21":
        params={"ins":"HG-4SAT-V150-C1350-21","N_steps":50,"replicas":50,"T_min":0.0001,"T_max":0.9,"c_grad":0.52,"lr":0.65,"alpha":0.09,"wd":0.02,"mom":0.62,"optimizer":"RMSprop"}
    elif instance == "HG-4SAT-V150-C1350-22":
        params={"ins":"HG-4SAT-V150-C1350-22","N_steps":50,"replicas":100,"T_min":0.0001,"T_max":0.76,"c_grad":0.67,"lr":0.62,"alpha":0.09,"wd":0.02,"mom":0.75,"optimizer":"RMSprop"}
    elif instance == "HG-4SAT-V150-C1350-23":
        params={"ins":"HG-4SAT-V150-C1350-23","N_steps":100,"replicas":100,"T_min":0.0001,"T_max":0.8,"c_grad":0.71,"lr":0.56,"alpha":0.1,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "HG-4SAT-V150-C1350-24":
        params={"ins":"HG-4SAT-V150-C1350-24","N_steps":100,"replicas":100,"T_min":0.0001,"T_max":0.76,"c_grad":0.85,"lr":0.56,"alpha":0.1,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "HG-4SAT-V150-C1350-3":
        params={"ins":"HG-4SAT-V150-C1350-3","N_steps":80,"replicas":100,"T_min":0.0001,"T_max":0.8,"c_grad":0.6,"lr":0.61,"alpha":0.1,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "HG-4SAT-V150-C1350-4":
        params={"ins":"HG-4SAT-V150-C1350-4","N_steps":80,"replicas":100,"T_min":0.0001,"T_max":0.8,"c_grad":0.57,"lr":0.56,"alpha":0.1,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "HG-4SAT-V150-C1350-5":
        params={"ins":"HG-4SAT-V150-C1350-5","N_steps":80,"replicas":100,"T_min":0.0001,"T_max":0.8,"c_grad":0.76,"lr":0.56,"alpha":0.1,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "HG-4SAT-V150-C1350-6":
        params={"ins":"HG-4SAT-V150-C1350-6","N_steps":80,"replicas":100,"T_min":0.0001,"T_max":0.8,"c_grad":0.6,"lr":0.56,"alpha":0.1,"wd":0.02,"mom":0.9,"optimizer":"RMSprop"}
    elif instance == "HG-4SAT-V150-C1350-7":
        params={"ins":"HG-4SAT-V150-C1350-7","N_steps":80,"replicas":100,"T_min":0.0001,"T_max":0.67,"c_grad":0.68,"lr":0.6,"alpha":0.1,"wd":0.02,"mom":0.7,"optimizer":"RMSprop"}
    elif instance == "HG-4SAT-V150-C1350-8":
        params={"ins":"HG-4SAT-V150-C1350-8","N_steps":80,"replicas":100,"T_min":0.0001,"T_max":0.67,"c_grad":0.71,"lr":0.69,"alpha":0.13,"wd":0.02,"mom":0.62,"optimizer":"RMSprop"}
    elif instance == "HG-4SAT-V150-C1350-9":
        params={"ins":"HG-4SAT-V150-C1350-9","N_steps":80,"replicas":100,"T_min":0.0001,"T_max":0.7,"c_grad":0.53,"lr":0.9,"alpha":0.11,"wd":0.03,"mom":0.75,"optimizer":"RMSprop"}
    else:
        raise ValueError('Unknown instance!')
    return params