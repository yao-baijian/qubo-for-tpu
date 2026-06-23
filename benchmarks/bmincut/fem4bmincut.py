import numpy as np
import pandas as pd
import torch
import time,math
import torch.nn.functional as Fun
from utils import load_matrix
import warnings
warnings.filterwarnings("ignore")
def argmax_cut(J,p):
    config = Fun.one_hot(p.argmax(dim=2),num_classes=p.shape[2]).to(J.dtype)
    return config, expected_cut(J, config) / 2 # ((config.permute([0,2,1]).reshape([-1,N]) @ J).reshape([-1,p.shape[-1],N]).permute([0,2,1]) * (1-config)).sum(2).sum(1)/2.0

def expected_cut(J,p):
    N = p.shape[0]
    trials = p.shape[1]
    q = p.shape[2]
    return ((torch.matmul(J,  p.view(N, trials * q)).view(N, trials, q)) * (1-p)).sum((0, 2))

class FEM_bmincut:
    def __init__(
            self, J, beta, learning_rate, replicas, c_grad, wd , beta1,beta2, dev='cuda', dtype=torch.float64,
            seed=1, q=2, fer=1.0, imba=5.0, h_factor=0.001):
        self.dtype = dtype
        self.dev = dev
        self.J = J.to(self.dtype).to(self.dev).to_sparse_coo()
        self.beta = torch.from_numpy(beta).to(self.dtype).to(self.dev) 
        self.N = self.J.shape[0]
        self.fer = fer
        self.replicas = replicas
        self.betas = (beta1,beta2)
        self.q = q
        d=torch.matmul(torch.abs(self.J),torch.ones([self.N,1],device=self.dev,dtype=self.dtype))
        d[d==0]=1
        self.c = (1/d).unsqueeze(2).expand(self.N,self.replicas,self.q) * self.q
        self.imba_c = (1/torch.sqrt(d)).unsqueeze(2).expand(self.N,self.replicas,self.q) *self.q
        self.imba_s = torch.linspace(0, imba, self.beta.shape[0]).to(self.dtype).to(self.dev)
        self.learning_rate = learning_rate
        self.seed = seed
        self.wd = wd
        self.c_grad = c_grad
        self.h_factor = h_factor
        if J.is_sparse:
            self.J = self.J.to_sparse_csr()
        self.initialize()
        
    def initialize(self):
        if self.seed is not None:
            torch.manual_seed(self.seed)
        self.h = self.h_factor * torch.randn([self.N,self.replicas,self.q], device=self.dev,dtype=self.dtype)
            
    def manual_grad_bmincut(self,step):
        self.e = Fun.one_hot(self.p.argmax(dim=2),num_classes=self.p.shape[2]).to(self.J.dtype)
        temp = torch.matmul(self.J,  1 - 2 * self.e.view(self.N, self.replicas * self.q)).view(self.N, self.replicas, self.q)
        s =  (2*(self.e.sum(0,keepdim=True) - self.e))
        tp = self.c_grad*((self.c * temp + self.imba_s[step] * self.imba_c * s) + (torch.log(self.p+1e-30) + 1)/self.beta[step])
        h_grad = (tp  - (tp * self.p).sum(2,keepdim=True).expand(tp.shape)) * self.p
        return h_grad
    
    def iterate(self):
        
        optimizer = torch.optim.Adam([self.h], lr=self.learning_rate,weight_decay=self.wd, betas=self.betas)
        for step in range(self.beta.shape[0]):
            """ unnormalized probabilities as a function of field self.h """
            self.p = torch.softmax(self.h, dim=2)
            grad = self.manual_grad_bmincut(step)
            self.h.grad = grad
            optimizer.step()
        
        return argmax_cut(self.J, self.p)

def banlanced_cut(config, cut, N, q, cutmax_const):
    max_group = config.sum(0).max(1)[0]
    imbalance = 0
    balance_constraint = (1 + 0.01 * imbalance) * np.ceil(N/q)

    pos = torch.where(max_group<=balance_constraint)[0]
    if len(pos)>0:
        cut_pos = torch.argmin(cut[pos])
        balanced_min_cut = cut[pos][cut_pos].item()
    else:
        balanced_min_cut = cutmax_const
    return balanced_min_cut, max_group.min().item()



def main(instance, q, dev, trials):
    if instance in ['add20','3elt','data','bcsstk33']:
        path = './real_world_graphs/'+instance +'.txt' 
    elif instance in ['N1000c5','N10000c5','N100000c5','N1000000c5']:
        path = './erdos_renyi_graphs/'+instance +'.txt' 
    else:
        raise ValueError('Unknown instance!')
    J, N, m = load_matrix(path,'torch','csr', return_n_m=True)
    params = load_params(instance, q)
    
    beta = np.exp(np.linspace(math.log(params['beta_min']),math.log(params['beta_max']),params['N_steps']))
    print(f'>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Processing graph from {instance}, {N} vertices, {m} edges, mode: bmincut <<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print(f"\n--------------------------------------------- {instance}: target group: {q} ---------------------------------------------")
    best_cut = m
    best_cut_seed = -1
    for trial in range(trials):
        print(f'\ntrial = {trial}, device = {dev}')
        seed = np.random.randint(0,10*trials,1)
        fem = FEM_bmincut(J, beta, params['lr'], params['replica'], params['c_grad'], params['wd'], params['beta1'], 
                            params['beta2'], dev=dev, dtype=torch.float32, seed=seed, q=q, fer=1, imba=params['imba'], h_factor=0.001)
        if dev != 'cpu':
            torch.cuda.synchronize(dev)
        start_t = time.perf_counter()
        config, cut = fem.iterate()
        if dev != 'cpu':
            torch.cuda.synchronize(dev)
        end_t = time.perf_counter()

        print("FEM:\tmin %.2f, max %.2f, mean %.2f, std %.2f" % (cut.min(), cut.max(), cut.mean(), cut.std()), 
            f" \tTime: {(end_t-start_t):.7f} Secs. for {params['replica']} replicas with {params['N_steps']} steps.")
        
        min_cut, min_max_group_size = banlanced_cut(config, cut, N, q, m)
        if min_cut < best_cut:
            best_cut = min_cut
            best_cut_seed = seed
        print(f"\nBalanced min-cut value with the ideal group size found by FEM: ", min_cut,' <<<<============ Cut of FEM. ')
        print(f"max_group_size: {min_max_group_size}.")
    print('best_cut:', best_cut)
    print('best_cut_seed:', best_cut_seed)
    

def load_params(instance, q):
    if instance == 'add20':
        if q == 2:
            params = {'q':2,'N_steps':10000,'replica':2000,'beta_min':0.0106,'beta_max':378,'c_grad':8.9,
                      'lr':0.2914, 'imba':1, 'wd':0.00034, 'beta1':0.9408,'beta2':0.7829,'optimizer':'adam'}
        elif q == 4:
            params = {'q':4,'N_steps':10000,'replica':2000,'beta_min':0.02527,'beta_max':420.42,'c_grad':6.664,
                      'lr':0.3664, 'imba':0.2332, 'wd':0.003411, 'beta1':0.9158,'beta2':0.7691,'optimizer':'adam'}
        elif q == 8:
            params = {'q':8, 'N_steps':10000,'replica':2000,'beta_min': 0.01524,'beta_max':1089.32,'c_grad':5.024,
                      'lr':0.4564, 'imba':0.6229, 'wd':0.0004198, 'beta1':0.9018,'beta2':0.7225,'optimizer':'adam'}
        elif q == 16:
            params = {'q':16, 'N_steps':10000,'replica':2000,'beta_min': 0.01817,'beta_max':803.72638,'c_grad':40.52,
                      'lr':0.6564, 'imba':1.0553, 'wd':0.008264, 'beta1':0.9032,'beta2':0.6009,'optimizer':'adam'}
        elif q == 32:
            params = {'q':32, 'N_steps':10000,'replica':2000,'beta_min': 0.01885,'beta_max':1413.72,'c_grad':3.6922,
                      'lr':1.4246, 'imba':1.9607, 'wd':0.01719, 'beta1':0.911,'beta2':0.8199,'optimizer':'adam'}
        else:
            raise ValueError('Unknown q value!')
        
    elif instance == 'data':
        if q == 2:
            params = {'q':2,'N_steps':10000,'replica':2000,'beta_min':0.02527,'beta_max':420.42,'c_grad': 6.664,
                      'lr':0.2629, 'imba':0.2292, 'wd':0.001706, 'beta1':0.9347,'beta2':0.7692,'optimizer':'adam'}
        elif q == 4:
            params = {'q':4,'N_steps':10000,'replica':2000,'beta_min':0.02527,'beta_max':420.42,'c_grad':6.664,
                      'lr':0.2629, 'imba':0.2292, 'wd':0.001706, 'beta1':0.9347,'beta2':0.7692,'optimizer':'adam'}
        elif q == 8:
            params = {'q':8, 'N_steps':10000,'replica':2000,'beta_min':0.02527,'beta_max':420.42,'c_grad':6.664,
                      'lr':0.3023, 'imba':0.3438, 'wd':0.001365, 'beta1':0.9369,'beta2':0.8076,'optimizer':'adam'}
        elif q == 16:
            params = {'q':16, 'N_steps':10000,'replica':2000,'beta_min': 0.03658,'beta_max':883.07,'c_grad':7.868,
                      'lr':0.8358, 'imba':1.3376, 'wd':0.0001874, 'beta1':0.7801,'beta2':0.4039,'optimizer':'adam'}
        elif q == 32:
            params = {'q':32, 'N_steps':12000,'replica':2000,'beta_min':0.009839,'beta_max':306.94,'c_grad':28.78,
                      'lr':0.6633, 'imba':0.697, 'wd':0.00165, 'beta1':0.5263,'beta2':0.4681,'optimizer':'adam'}
        else:
            raise ValueError('Unknown q value!')
        
    elif instance == '3elt':
        if q == 2:
            params = {'q':2,'N_steps':10000,'replica':2000,'beta_min':0.01824,'beta_max':1229.32,'c_grad':3.573,
                      'lr':1.2032, 'imba':2.07, 'wd':0.02293, 'beta1':0.9374,'beta2':0.9215,'optimizer':'adam'}
        elif q == 4:
            params = {'q':4,'N_steps':10000,'replica':2000,'beta_min':0.01824,'beta_max':1229.32,'c_grad':3.573,
                      'lr':1.2032, 'imba':2.07, 'wd':0.02293, 'beta1':0.9374,'beta2':0.9215,'optimizer':'adam'}
        elif q == 8:
            params = {'q':8, 'N_steps':10000,'replica':2000,'beta_min':0.01459,'beta_max':983.46,'c_grad':3.692,
                      'lr':0.2117, 'imba':0.8, 'wd':0.001031, 'beta1':0.724,'beta2':0.5397,'optimizer':'adam'}
        elif q == 16:
            params = {'q':16, 'N_steps':10000,'replica':2000,'beta_min':0.01824 ,'beta_max':1229.32,'c_grad':3.573,
                      'lr':0.2238, 'imba':1.267, 'wd':0.002577, 'beta1':0.77,'beta2':0.7698,'optimizer':'adam'}
        elif q == 32:
            params = {'q':32, 'N_steps':12000,'replica':2000,'beta_min':0.02313,'beta_max':1790.71,'c_grad':3.384,
                      'lr':0.9946, 'imba':1.369, 'wd':0.02508, 'beta1':0.777,'beta2':0.8982,'optimizer':'adam'}
        else:
            raise ValueError('Unknown q value!')
    elif instance == 'bcsstk33':
        if q == 2:
            params = {'q':2,'N_steps':10000,'replica':2000,'beta_min':0.01824,'beta_max':1229.32,'c_grad':3.573,
                      'lr':1.2032, 'imba':2.07, 'wd':0.022926, 'beta1':0.9374,'beta2':0.92151,'optimizer':'adam'}
        elif q == 4:
            params = {'q':4,'N_steps':10000,'replica':2000,'beta_min':0.01824,'beta_max':1229.32,'c_grad':3.573,
                      'lr':1.2032, 'imba':2.07, 'wd':0.022926, 'beta1':0.9374,'beta2':0.92151,'optimizer':'adam'}
        elif q == 8:
            params = {'q':8, 'N_steps':12000,'replica':2000,'beta_min':0.0186,'beta_max':377.11,'c_grad':7.8,
                      'lr':0.1948, 'imba':0.5016, 'wd':0.001663, 'beta1':0.7196,'beta2':0.9,'optimizer':'adam'}
        elif q == 16:
            params = {'q':16, 'N_steps':12000,'replica':2000,'beta_min':0.01668 ,'beta_max':475,'c_grad':3.526,
                      'lr':0.5254, 'imba':1.167, 'wd':0.003089, 'beta1':0.8894,'beta2':0.6223,'optimizer':'adam'}
        elif q == 32:
            params = {'q':32, 'N_steps':12000,'replica':2000,'beta_min':0.01459,'beta_max':1413.72,'c_grad':3.692,
                      'lr':1.5541, 'imba':1.656, 'wd':0.01146, 'beta1':0.9582,'beta2':0.8686,'optimizer':'adam'}
        else:
            raise ValueError('Unknown q value!')
    elif instance == 'N1000c5':
        if q == 4:
            params = {'q':4,'N_steps':12000,'replica':50,'beta_min':0.02582,'beta_max':2179.27,'c_grad':23.695,
                      'lr':0.228, 'imba':0.1228, 'wd':0.003836, 'beta1':0.7123,'beta2':0.4987,'optimizer':'adam'}
        elif q == 8:
            params = {'q':8, 'N_steps':12000,'replica':50,'beta_min':0.02466,'beta_max':2960.97,'c_grad':24.925,
                      'lr':0.3128, 'imba':0.1066, 'wd':0.00341, 'beta1':0.8076,'beta2':0.9999,'optimizer':'adam'}
        elif q == 16:
            params = {'q':16, 'N_steps':12000,'replica':50,'beta_min':0.0208 ,'beta_max':2664.61,'c_grad':15.7,
                      'lr':0.131, 'imba':0.2062, 'wd':0.00213, 'beta1':0.9229,'beta2':0.9999,'optimizer':'adam'}
        else:
            raise ValueError('Unknown q value!')
    elif instance == 'N10000c5':
        if q == 4:
            params = {'q':4,'N_steps':12000,'replica':50,'beta_min':0.009694,'beta_max':2810.53,'c_grad':24.2,
                      'lr':0.5273, 'imba':0.03872, 'wd':0.0005121, 'beta1':0.909,'beta2':0.8273,'optimizer':'adam'}
        elif q == 8:
            params = {'q':8, 'N_steps':12000,'replica':50,'beta_min':0.006024,'beta_max':834.533,'c_grad':260.59,
                      'lr':0.3057, 'imba':0.0867, 'wd':0.1549, 'beta1':0.9091,'beta2':0.842,'optimizer':'adam'}
        elif q == 16:
            params = {'q':16, 'N_steps':12000,'replica':50,'beta_min':0.01002 ,'beta_max':5934.19,'c_grad':1218.1,
                      'lr':0.6402, 'imba':0.01659, 'wd':0.00953, 'beta1':0.7142,'beta2':0.9999,'optimizer':'adam'}
        else:
            raise ValueError('Unknown q value!')
    elif instance == 'N100000c5':
        if q == 4:
            params = {'q':4,'N_steps':12000,'replica':50,'beta_min':0.0457,'beta_max':12038.97,'c_grad':392.86,
                      'lr':1.539, 'imba':0.005964, 'wd':2.4795, 'beta1':0.9091,'beta2':0.8977,'optimizer':'adam'}
        elif q == 8:
            params = {'q':8, 'N_steps':12000,'replica':50,'beta_min':0.00995,'beta_max':6730.87,'c_grad':1808.42,
                      'lr':0.6751, 'imba':0.00239, 'wd':0.572, 'beta1':0.834,'beta2':0.9999,'optimizer':'adam'}
        elif q == 16:
            params = {'q':16, 'N_steps':12000,'replica':50,'beta_min': 0.00897,'beta_max':381.074,'c_grad':41.92,
                      'lr':0.2932, 'imba':0.087, 'wd':0.0141, 'beta1':0.9614,'beta2':0.9999,'optimizer':'adam'}
        else:
            raise ValueError('Unknown q value!')
    elif instance == 'N1000000c5':
        if q == 4:
            params = {'q':4,'N_steps':12000,'replica':50,'beta_min':0.02274,'beta_max':4477.37,'c_grad':30.0353,
                      'lr':1.834, 'imba':0.00182, 'wd':1.097, 'beta1':0.9615,'beta2':0.99999,'optimizer':'adam'}
        elif q == 8:
            params = {'q':8, 'N_steps':12000,'replica':50,'beta_min':0.007268,'beta_max':3996.4,'c_grad':666.5,
                      'lr':0.2348, 'imba':0.0003377, 'wd':0.1952, 'beta1':0.715,'beta2':0.6972,'optimizer':'adam'}
        elif q == 16:
            params = {'q':16, 'N_steps':12000,'replica':50,'beta_min': 0.00299,'beta_max':7459.056,'c_grad':306.38,
                      'lr':0.2862, 'imba':0.00767, 'wd':0.5576, 'beta1':0.9614,'beta2':0.8968,'optimizer':'adam'}
        else:
            raise ValueError('Unknown q value!')
    else:
        raise ValueError('Unknown instance!')
    return params