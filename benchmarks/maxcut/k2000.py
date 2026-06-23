import numpy as np
import torch
import pandas as pd
import time
import torch.nn.functional as Fun
from mpl_toolkits.axes_grid1 import host_subplot
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
import warnings
warnings.filterwarnings('ignore')

def load_matrix(path:'str',numer_package:'str',store_format:'str',return_n_m:'bool'= False) -> 'float':
    """"
    Load the coupling matrix of the Graph instance from '.txt' file to python matrix.
    Example:  xxx.txt
        730 31147   first line format: (total number of nodes) (total number of edges) 
        0 1 1       other lines format: (node1) (node2) [edge weight]
        0 3 1                           1. the ordinal number(or index) for nodes starting from 0 or 1.
        0 4 1                           2. edge weights by defualt setting to 1 if unassigned.
        0 5 1
        0 6 1
        ...
    Parameters:
    :param path - The file path of the graph instance, with format of '.txt';
    :param numer_package - Choose the preferred python sci-package, choices = 'scipy' and 'torch';
    :param store_format - The output matrix will be with the store format of 'store_format', choices = 'csr', 'csc' and 'dense'.
    Returns:
    The output coupling matrix of the instance graph.
    """ 
    with open(path, "r") as f:
        l = f.readline()
        N, edges = [int(x) for x in l.split(" ") if x != "\n"]
        
    G = pd.read_csv(path,sep=' ',skiprows=[0],index_col=False, header=None,names=['node1','node2','weight'])
    G.fillna({'weight':float(1.0)},inplace=True)
    shift = G.iloc[0,0]
    ori_graph = np.array([list(np.concatenate([G.iloc[:,0]-shift,G.iloc[:,1]- shift])),
                 list(np.concatenate([G.iloc[:,1]-shift,G.iloc[:,0]- shift])),
                 list(np.concatenate([G.iloc[:,-1],G.iloc[:,-1]]))])
    ori_graph = ori_graph.T[np.lexsort((ori_graph[1,:],ori_graph[0,:])).tolist()].T
    if numer_package == 'scipy':
        J = coo_matrix((ori_graph[2,:].tolist(),
                             (ori_graph[0,:].tolist(),ori_graph[1,:].tolist())), shape=(N, N))
        if J.shape[0] != N:
            raise ValueError("The shape of J does not match N!")
        if J.data.shape[0]/2 != edges:
            raise ValueError("The number of elements in J does not match edges!")
        if store_format == 'csr':
            J = csr_matrix(J)
        elif store_format == 'csc':
            J = csc_matrix(J)
        elif store_format == 'dense':
            J = J.todense()
        else:
            raise ValueError("Error: Input wrong 'store_format'! Please choose from ['csr', 'csc', 'dense'].")
    elif numer_package == 'torch':
        J = torch.sparse_coo_tensor([ori_graph[0,:].tolist(),
                                 ori_graph[1,:].tolist()], 
                                    ori_graph[2,:].tolist(),(N, N))
        if J.shape[0] != N:
            raise ValueError("The shape of J does not match N!")
        if J._values().shape[0]/2 != edges:
            raise ValueError("The number of elements in J does not match edges!")
        
        if store_format == 'csr':
            J = J.to_sparse_csr()
        elif store_format == 'csc':
            J = J.to_sparse_csc()
        elif store_format == 'dense':
            J = J.to_dense()
        else:
            raise ValueError("Error: Input wrong 'store_format'! Please choose from ['csr', 'csc', 'dense'].")
    else:
        raise ValueError("Error: Input wrong 'numer_package'! Please choose from ['scipy', 'torch'].")
    if return_n_m is False:
        return J.float()
    else:
        return J.float(), N, edges
    
class SBM():
    def __init__(self, J, N_step, dt, trials, dev='cuda', seed = -1, dtype=torch.float32):
        self.dtype = dtype 
        self.dev = dev
        self.N = J.shape[0]
        self.J = J.to(self.dtype).to(self.dev)
        self.c0 = 0.5/((self.N**0.5)*((torch.sum(J*J))/(self.N*(self.N-1)))**0.5)
        self.N_step = N_step
        self.dt = dt
        self.sum_w = torch.sum(-self.J)/4
        self.trials = trials
        self.seed = seed    #  seed < 0 means seed is disabled.
        self.a = torch.linspace(0, 1, self.N_step)
        # self.a[self.a>1] = 1
        self.a = self.a.to(self.dev).to(self.dtype)
        
    def calculate_cut(self):
        cut = self.sum_w + 0.25 * torch.sum(torch.sign(self.x) * 
                                       torch.matmul(self.J,torch.sign(self.x)), 0)
        return cut  
    
    def iterate(self):
        if self.seed >= 0:
            torch.manual_seed(self.seed)
        self.x = (torch.rand([self.N, self.trials],device = self.dev,dtype=self.dtype)-0.5) * 0.2
        if self.seed >= 0:
            torch.manual_seed(self.seed + 30)
        self.y = (torch.rand([self.N, self.trials],device = self.dev,dtype=self.dtype)-0.5) * 0.2
        self.run_time = []
        # self.cut = torch.zeros([self.trials,self.N_step]).to(self.dev)
        
        for jj in range(self.N_step):
            # self.cut[:,jj] = self.calculate_cut()
            t_start = time.perf_counter()
            z = self.c0 * torch.matmul(self.J,torch.sign(self.x))
            self.y = self.y + (-(1-self.a[jj]) * self.x + z) * self.dt
            self.x = self.x + self.y * self.dt
            self.y = torch.where(torch.abs(self.x) > 1, 0, self.y)
            # self.x = torch.where(torch.abs(self.x) > 1, torch.sign(self.x), self.x)
            self.x = torch.where(self.x > 1, 1, self.x)
            self.x = torch.where(self.x < -1, -1, self.x)
            t_end = time.perf_counter()
            self.run_time.append(t_end - t_start)


def argmax_cut(J,p):
    config = (p > 0.5).to(J.dtype)
    return config, expected_cut(J, config) / 2
def expected_cut(J,p):
    return 2 * ((torch.matmul(J,p) * (1-p))).sum(0)

class FEM:
    def __init__(
            self, J, beta, learning_rate, replicas,c_grad, dev='cuda', dtype=torch.float32,
            seed=-1, wd = 0.01, alpha= 0.98, mom=0.91, q=2, h_factor=0.01
        ):
        self.dtype = dtype
        self.dev = dev
        self.J = J.to_dense().to(self.dtype).to(self.dev)
        self.beta = torch.from_numpy(beta).to(self.dtype).to(self.dev)
        self.N = self.J.shape[0]
        self.replicas = replicas
        # self.c = 0.5/((self.N**0.5)*((torch.sum(self.J*self.J))/(self.N*(self.N-1)))**0.5)
        self.c = 1/torch.abs(self.J).sum(1,keepdim=True)
        self.c_grad = c_grad
        self.learning_rate = learning_rate
        self.seed = seed
        self.q = q
        self.alpha = alpha
        self.wd = wd
        self.mom = mom
        self.h_factor = h_factor
        self.J1 = 0.25*self.c * self.J
        if J.is_sparse:
            self.J1 = self.J1.to_sparse_csr()
        self.initialize()
    
    def manual_grad_maxcutv4(self, beta):
        h_grad = self.c_grad * torch.addmm(self.h,self.J1,torch.sign(self.m),beta = 0.25/beta) * (1-self.m**2)
        return h_grad
    
    def manual_grad_maxcutv2(self, beta):
        h_grad = self.c_grad * torch.addmm(self.h,self.J1,torch.sign(self.m),beta = 0.25/beta) * (1-self.m**2)
        return h_grad
    def initialize(self):
        if self.seed > 0:
            torch.manual_seed(self.seed)
        self.h = self.h_factor * torch.randn([self.N, self.replicas], device=self.dev,dtype=self.dtype)
        self.optimizer = torch.optim.RMSprop([self.h], lr=self.learning_rate, alpha=self.alpha, eps=1e-08, weight_decay=self.wd, momentum=self.mom, centered=False)
        # self.optimizer = torch.optim.SGD([self.h], lr=self.learning_rate,momentum=self.mom, alphaing=self.alpha,weight_decay=self.wd)

    def iterate(self):
        self.run_time = []
        for step in range(self.beta.shape[0]):
            t_start = time.perf_counter()
            self.m = torch.tanh(self.h/2)
            self.h.grad = self.manual_grad_maxcutv2(self.beta[step])
            self.optimizer.step()
            t_end = time.perf_counter()
            self.run_time.append(t_end - t_start)
            
    def calculate_results(self):
        self.p = torch.sigmoid(self.h)
        return argmax_cut(self.J, self.p)
    
def load_params(algorithm):
    if algorithm == 'dsb':
        params = {'dt':1.25, 'seed':[7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7]}
    elif algorithm == 'fem':
        params = {'lr':0.03, 'wd': 0.013, 'alpha' : 0.56, 'Tmax': 1.16,
           'Tmin': 6e-05, 'c_grad': 1, 'mom': 0.63, 'h_factor':0.001, 
           'seed':[4,3,3,16,3,9,16,4,6,6,8,14,3,3,3,3]}
    else:
        raise ValueError("enter 'dsb' or 'fem' ")
    return params