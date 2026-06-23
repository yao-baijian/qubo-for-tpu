import torch,re,time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
sys_path = '.'
def load_matrix(path:'str', return_n_m:'bool'=False) -> 'float':
    with open(path, "r") as f:
        l = f.readline()
        N, edges = [int(x) for x in l.split(" ") if x != "\n"]
    G = pd.read_csv(path, sep=' ',skiprows=[0],index_col=False, header=None,names=['node1','node2','weight'])
    G.fillna({'weight':float(1.0)},inplace=True)
    shift = G.iloc[0,0]
    ori_graph = np.array([list(np.concatenate([G.iloc[:,0]-shift,G.iloc[:,1]- shift])),
                 list(np.concatenate([G.iloc[:,1]-shift,G.iloc[:,0]- shift])),
                 list(np.concatenate([G.iloc[:,-1],G.iloc[:,-1]]))])
    ori_graph = ori_graph.T[np.lexsort((ori_graph[1,:],ori_graph[0,:])).tolist()].T
    J = torch.sparse_coo_tensor([ori_graph[0,:].tolist(),
                                ori_graph[1,:].tolist()], 
                                ori_graph[2,:].tolist(),(N, N))
    if J.shape[0] != N:
        raise ValueError("The shape of J does not match N!")
    if J._values().shape[0]/2 != edges:
        raise ValueError("The number of elements in J does not match edges!")
    if return_n_m is False:
        return J
    else:
        return J.float().to_sparse_csr(), N, edges
def argmax_cut(J,p):
    config = (p > 0.5).to(J.dtype)
    return config, expected_cut(J, config) / 2
def expected_cut(J,p):
    return 2 * ((torch.matmul(J,p) * (1-p))).sum(0)
class FEM_4_MaxCut:
    def __init__(
            self, J, beta, learning_rate, replicas, c_grad, dev='cuda:0', dtype=torch.float32,
            seed=-1, h_factor=0.001, optimizer = 'rmsprop', params = None
        ):
        self.dtype,self.dev,self.replicas = dtype,dev,replicas
        self.J = J.to_dense().to(self.dtype).to(self.dev)
        self.beta = torch.from_numpy(beta).to(self.dtype).to(self.dev)
        self.N = self.J.shape[0]
        self.c = 1/torch.abs(self.J).sum(1,keepdim=True)
        self.c_grad,self.h_factor,self.seed = c_grad,h_factor,seed
        self.learning_rate = learning_rate
        self.J1 = 0.25 * self.c * self.J
        if J.is_sparse:
            self.J1 = self.J1.to_sparse_csr()
        self.opt_mode = optimizer
        self.params = params
        self.initialize()
    def initialize(self):
        if self.seed > 0:
            torch.manual_seed(self.seed)
        self.h = self.h_factor * torch.randn([self.N,self.replicas], device=self.dev,dtype=self.dtype)
        if self.params is None:
            raise ValueError
        else:
            if self.opt_mode == 'rmsprop':
                alpha, wd, mom = self.params
                self.optimizer = torch.optim.RMSprop([self.h], lr=self.learning_rate, alpha=alpha, weight_decay=wd, momentum=mom)
            elif self.opt_mode == 'sgd':
                dampen, wd, mom = self.params
                self.optimizer = torch.optim.SGD([self.h], lr=self.learning_rate,momentum=mom, dampening=dampen,weight_decay=wd)
            else:
                raise ValueError
    def manual_grad_maxcut(self, beta):
        h_grad = self.c_grad * torch.addmm(self.h,self.J1,torch.sign(self.m),beta = 0.25/beta) * (1-self.m**2)
        return h_grad
    def iterate(self):
        for step in range(self.beta.shape[0]):
            self.m = torch.tanh(self.h/2)
            self.h.grad = self.manual_grad_maxcut(self.beta[step])
            self.optimizer.step()
    def calculate_results(self):
        self.p = torch.sigmoid(self.h)
        return argmax_cut(self.J, self.p)
    
def calculate_ps_tts_ttt(instance, cut, N_rep, N_batch):
    TTS_target,TTT_target = TTS_TTT_target(instance)
    count_TTS,max_found, count_TTT = 0,0,0
    for ii in range(N_rep):
        max_value = cut[ii * N_batch : (ii+1) * N_batch].max()
        if max_value>max_found:
            max_found = max_value.item()
        if max_value >= TTS_target:
            count_TTS += 1
        if max_value >= TTT_target:
            count_TTT += 1
    P_s_TTS = count_TTS / N_rep
    P_s_TTT = count_TTT / N_rep
    return P_s_TTS, P_s_TTT, max_found

def TTS_TTT_target(instance):
    with open(sys_path+'/targetvalue.txt', 'r', encoding='utf-8') as f:
        content = f.read()
        result = re.findall(".*"+instance+" (.*).*", content)
        target_value = int(result[0]) if result else 0
        TTS_target = float(target_value)
        TTT_target = float(target_value)*0.99
        return TTS_target, TTT_target

def TTS_TTT(T_com, P_s):
    return T_com * np.log(1-0.99)/np.log(1-P_s) if P_s < 0.99 else T_com

def maxcut_print_results(
    instance, max_found, N, seed, dev, T_com, P_s_TTS, P_s_TTT, N_rep, 
    N_batch, N_step, C_grad, lr, Tmin, Tmax, opt_mode, params,dsb_tts):

    print(f"========= mode: MaxCut =========")
    print("instance: ", instance)
    print(f'num. of nodes: {N},  seed = {seed}, device = {dev}\n')
    TTS_target, TTT_target = TTS_TTT_target(instance)
    print('TTS_target:', TTS_target)
    print('TTT_target:', TTT_target)
    print(f"T_com: {T_com :.2f} ms")
    if P_s_TTS>0:
        print('P_s_TTS: ', P_s_TTS)
        print(f"TTS of FEM: {TTS_TTT(T_com, P_s_TTS):.2f} ms, TTS of dSB: {dsb_tts} ms")
    else:
        print("P_s_TTS is zero!")
    if P_s_TTT>0:
        print('P_s_TTT: ', P_s_TTT)
        print(f"TTT: {TTS_TTT(T_com, P_s_TTT):.2f} ms")    
    else:
        print("P_s_TTT is zero!")
    if opt_mode == 'rmsprop':
        alpha, wd, mom = params
        print(f"Params: N_rep: {N_rep}, N_batch: {N_batch}, N_step: {N_step}, C_grad: {C_grad}, lr: {lr}, Tmin:{Tmin}, Tmax: {Tmax}, " + 
                f"alpha: {alpha}, weight_decay: {wd}, mom:{mom}.")
    elif opt_mode == 'sgd':
        dampen, wd, mom = params
        print(f"Params: N_rep: {N_rep}, N_batch: {N_batch}, N_step: {N_step}, C_grad: {C_grad}, lr: {lr}, Tmin:{Tmin}, Tmax: {Tmax}, " + 
                f"\ndampen: {dampen}, weight_decay: {wd}, mom:{mom}.")
    
    print('Maximum cut value found: ', int(max_found))
    print(f'Best known result of {instance} is {int(TTS_target)}')


def main(instance,dev):
    J, N, m = load_matrix(sys_path+'/Gset/'+ instance,return_n_m=True)
    params_dic = load_params(instance)
    beta = 1/np.linspace(params_dic['Tmax'], params_dic['Tmin'], params_dic['N_step'])
    replicas = params_dic['N_batch'] * params_dic['N_rep']
    fem = FEM_4_MaxCut(J, beta, params_dic['lr'], replicas, params_dic["C_grad"], dev=dev, dtype=torch.float32,
            seed=params_dic["seed"], h_factor=0.001, optimizer = params_dic["opt_mode"], params = params_dic["params"])
    
    torch.cuda.synchronize(dev)
    start_t = time.perf_counter()
    fem.iterate()
    torch.cuda.synchronize(dev)
    end_t = time.perf_counter()
    config, cut = fem.calculate_results()
    T_com = (float(end_t-start_t)*1000)/(params_dic['N_rep'])
    P_s_TTS, P_s_TTT, max_found = calculate_ps_tts_ttt(instance, cut, params_dic['N_rep'], params_dic['N_batch'])
    maxcut_print_results(
    instance, max_found, N, params_dic["seed"], dev, T_com, P_s_TTS, P_s_TTT, params_dic['N_rep'],
    params_dic['N_batch'], params_dic['N_step'], params_dic["C_grad"], params_dic['lr'],
    params_dic['Tmin'], params_dic['Tmax'], params_dic["opt_mode"], params_dic["params"],params_dic["dsb_tts"])
    print("FEM: min %.2f, max %.2f, mean %.2f, std %.2f" % (cut.min(), cut.max(), cut.mean(), cut.std()))
    
def load_params(instance):
    if instance == 'G1':
        params_dic = {'N_batch':130,'N_step':1000, 'N_rep':1000, "C_grad":1, 'seed': 92, 'lr':0.2, 
                        'Tmin':8.00E-05, 'Tmax':0.5,'opt_mode':'rmsprop', 'params':[0.623,0.02,0.693],'dsb_tts':33.3}
    if instance == 'G2':
        params_dic = {'N_batch':100,'N_step':5000, 'N_rep':1000, "C_grad":1, 'seed': 70, 'lr':0.0717,
                        'Tmin':6.34E-04, 'Tmax':0.2592,'opt_mode':'rmsprop', 'params':[0.5485,0.0264,0.9082],'dsb_tts':239}
    if instance == 'G3':
        params_dic = {'N_batch':120,'N_step':1000, 'N_rep':1000, "C_grad":1, 'seed': 97, 'lr':0.3174,
                        'Tmin':1.10E-03, 'Tmax':0.264,'opt_mode':'rmsprop', 'params':[0.7765,0.00672,0.7804],'dsb_tts':46.2}
    if instance == 'G4':
        params_dic = {'N_batch':130,'N_step':800, 'N_rep':1000, "C_grad":1, 'seed': 30, 'lr':0.2691,
                        'Tmin':8.90E-04, 'Tmax':0.29,'opt_mode':'rmsprop', 'params':[0.4718,0.00616,0.7414],'dsb_tts':34.4}
    if instance == 'G5':
        params_dic = {'N_batch':110,'N_step':1000, 'N_rep':1000, "C_grad":1, 'seed': 86, 'lr':0.24,
                        'Tmin':9.00E-04, 'Tmax':0.2,'opt_mode':'rmsprop', 'params':[0.9999,0.0056,0.8215],'dsb_tts':58.6}
    if instance == 'G6':
        params_dic = {'N_batch':20,'N_step':1000, 'N_rep':1000, "C_grad":1, 'seed': 72, 'lr':0.534,
                        'Tmin':1.70E-03, 'Tmax':0.5,'opt_mode':'rmsprop', 'params':[0.6045,0.00657,0.4733],'dsb_tts':6.3}
    if instance == 'G7':
        params_dic = {'N_batch':80,'N_step':700, 'N_rep':1000, "C_grad":1, 'seed': 21, 'lr':0.452, 
                        'Tmin':1.80E-03, 'Tmax':0.54,'opt_mode':'rmsprop', 'params':[0.8966,0.0087,0.632],'dsb_tts':6.85}
    if instance == 'G8':
        params_dic = {'N_batch':100,'N_step':1000, 'N_rep':1000, "C_grad":1, 'seed': 45, 'lr':0.296, 
                        'Tmin':7.92E-04, 'Tmax':0.19,'opt_mode':'rmsprop', 'params':[0.9999,0.00731,0.737],'dsb_tts':11.9}
    if instance == 'G9':
        params_dic = {'N_batch':70,'N_step':2500, 'N_rep':1000, "C_grad":1, 'seed': 73, 'lr':0.305, 
                        'Tmin':9.00E-04, 'Tmax':0.208,'opt_mode':'rmsprop', 'params':[0.9999,0.00205,0.718],'dsb_tts':36}
    if instance == 'G10':
        params_dic = {'N_batch':100,'N_step':2000, 'N_rep':1000, "C_grad":0.75, 'seed': 22, 'lr':1.2, 
                        'Tmin':5.21E-06, 'Tmax':1.28,'opt_mode':'sgd', 'params':[0.082,0.03,0.88],'dsb_tts':47.7}
    if instance == 'G11':
        params_dic = {'N_batch':120,'N_step':1800, 'N_rep':1000, "C_grad":0.98, 'seed': 53, 'lr':1.2, 
                        'Tmin':4.96E-06, 'Tmax':1.28,'opt_mode':'sgd', 'params':[0.13,0.061,0.88],'dsb_tts':3.49}
    if instance == 'G12':
        params_dic = {'N_batch':140,'N_step':1600, 'N_rep':1000, "C_grad":0.65, 'seed': 38, 'lr':1.98, 
                        'Tmin':7.80E-06, 'Tmax':1.28,'opt_mode':'sgd', 'params':[0.13,0.06,0.88],'dsb_tts':5.16}
    if instance == 'G13':
        params_dic = {'N_batch':130,'N_step':3000, 'N_rep':1000, "C_grad":1.7, 'seed': 87, 'lr':3, 
                        'Tmin':3.12E-06, 'Tmax':1.28,'opt_mode':'sgd', 'params':[0.082,0.033,0.76],'dsb_tts':11.9}
    if instance == 'G14':
        params_dic = {'N_batch':250,'N_step':7000, 'N_rep':1000, "C_grad":1, 'seed': 91, 'lr':0.44, 
                        'Tmin':8.64E-04, 'Tmax':0.387,'opt_mode':'rmsprop', 'params':[0.9999,0.0089,0.793],'dsb_tts':71633}
    if instance == 'G15':
        params_dic = {'N_batch':200,'N_step':4000, 'N_rep':1000, "C_grad":1, 'seed': 8, 'lr':0.45, 
                        'Tmin':1.00E-03, 'Tmax':0.5,'opt_mode':'rmsprop', 'params':[0.9999,0.0056,0.7327],'dsb_tts':340}
    if instance == 'G16':
        params_dic = {'N_batch':160,'N_step':7000, 'N_rep':1000, "C_grad":1, 'seed': 62, 'lr':0.288, 
                        'Tmin':8.10E-04, 'Tmax':0.54,'opt_mode':'rmsprop', 'params':[0.9999,0.00756,0.7877],'dsb_tts':347}
    if instance == 'G17':
        params_dic = {'N_batch':200,'N_step':7000, 'N_rep':1000, "C_grad":1, 'seed': 89, 'lr':0.631, 
                        'Tmin':1.06E-03, 'Tmax':0.253,'opt_mode':'rmsprop', 'params':[0.9999,0.01341,0.7642],'dsb_tts':1631}
    if instance == 'G18':
        params_dic = {'N_batch':150,'N_step':1000, 'N_rep':1000, "C_grad":1, 'seed': 58, 'lr':0.345, 
                        'Tmin':1.00E-03, 'Tmax':0.4,'opt_mode':'rmsprop', 'params':[0.99,0.01,0.99],'dsb_tts':375}
    if instance == 'G19':
        params_dic = {'N_batch':85,'N_step':1700, 'N_rep':1000, "C_grad":1.75, 'seed': 23, 'lr':4.368, 
                        'Tmin':3.98E-06, 'Tmax':0.962,'opt_mode':'sgd', 'params':[0.05175,0.01336,0.729],'dsb_tts':17.8}
    if instance == 'G20':
        params_dic = {'N_batch':100,'N_step':550, 'N_rep':1000, "C_grad":1.55, 'seed': 29, 'lr':1.38, 
                        'Tmin':9.40E-04, 'Tmax':0.37,'opt_mode':'rmsprop', 'params':[0.9089,0.00445,0.8186],'dsb_tts':9.02}
    if instance == 'G21':
        params_dic = {'N_batch':40 ,'N_step': 1000, 'N_rep':1000, "C_grad":1, 'seed': 51, 'lr':0.33, 
                        'Tmin':9.60E-04, 'Tmax':0.6,'opt_mode':'rmsprop', 'params':[0.9999,0.0092,0.692],'dsb_tts':260}
    if instance == 'G22':
        params_dic = {'N_batch':90 ,'N_step': 4700, 'N_rep':1000, "C_grad":1, 'seed': 59, 'lr':0.481, 
                        'Tmin':0.00024, 'Tmax':0.352,'opt_mode':'rmsprop', 'params':[0.9999,0.00382,0.7166],'dsb_tts':429}
    if instance == 'G23':
        params_dic = {'N_batch':10,'N_step': 3200, 'N_rep':1000, "C_grad":52.72, 'seed': 60, 'lr':8.042,
                        'Tmin':1.15E-06, 'Tmax':0.406,'opt_mode':'sgd', 'params':[0.1443,0.00184,0.714],'dsb_tts':89}
    if instance == 'G24':
        params_dic = {'N_batch':250 ,'N_step': 7000, 'N_rep':1000, "C_grad":1, 'seed': 26, 'lr':0.39, 
                        'Tmin':1.60E-04, 'Tmax':0.528,'opt_mode':'rmsprop', 'params':[0.9999,0.00413,0.74],'dsb_tts':459}
    if instance == 'G25':
        params_dic = {'N_batch':200 ,'N_step': 7000, 'N_rep':1000, "C_grad":5.33, 'seed': 57, 'lr':3.66, 
                        'Tmin':4.83E-06, 'Tmax':0.4,'opt_mode':'sgd', 'params':[0.0905,0.00987,0.672],'dsb_tts':2279}
    if instance == 'G26':
        params_dic = {'N_batch':200 ,'N_step': 6000, 'N_rep':1000, "C_grad":2.18, 'seed': 20, 'lr':8.46, 
                        'Tmin':4.43E-06, 'Tmax':0.361,'opt_mode':'sgd', 'params':[0.0612,0.0078,0.714],'dsb_tts':476}
    if instance == 'G27':
        params_dic = {'N_batch':80 ,'N_step': 2000, 'N_rep':1000, "C_grad":1, 'seed': 58, 'lr':0.7, 
                        'Tmin':5.00E-04, 'Tmax':0.28,'opt_mode':'rmsprop', 'params':[0.9995,0.00575,0.78],'dsb_tts':49.9}
    if instance == 'G28':
        params_dic = {'N_batch':100 ,'N_step': 3000, 'N_rep':1000, "C_grad":1, 'seed': 84, 'lr':0.69, 
                        'Tmin':5.00E-04, 'Tmax':0.32,'opt_mode':'rmsprop', 'params':[0.999,0.006,0.78],'dsb_tts':87.2}
    if instance == 'G29':
        params_dic = {'N_batch':120 ,'N_step': 4000, 'N_rep':1000, "C_grad":1, 'seed': 48, 'lr':0.44, 
                        'Tmin':2.70E-04, 'Tmax':0.38,'opt_mode':'rmsprop', 'params':[0.99991,0.013,0.7],'dsb_tts':221}
    if instance == 'G30':
        params_dic = {'N_batch':100 ,'N_step': 7000, 'N_rep':1000, "C_grad":1.9, 'seed': 46, 'lr':2.59, 
                        'Tmin':4.92E-06, 'Tmax':0.96,'opt_mode':'sgd', 'params':[0.05,0.053,0.715],'dsb_tts':439}
    if instance == 'G31':
        params_dic = {'N_batch':100 ,'N_step': 7000, 'N_rep':1000, "C_grad":1.32, 'seed': 49, 'lr':1.38, 
                        'Tmin':2.76E-06, 'Tmax':1.834,'opt_mode':'sgd', 'params':[0.0104,0.083,0.7566],'dsb_tts':1201}
    if instance == 'G32':
        params_dic = {'N_batch':20 ,'N_step': 12000, 'N_rep':1000, "C_grad":3.17, 'seed': 71, 'lr':1.67, 
                        'Tmin':1.42E-05, 'Tmax':0.89,'opt_mode':'sgd', 'params':[0.1285,0.018,0.9],'dsb_tts':3622}
    if instance == 'G33':
        params_dic = {'N_batch':260 ,'N_step': 12000, 'N_rep':1000, "C_grad": 2, 'seed': 50, 'lr':4.05, 
                        'Tmin':7.80E-06, 'Tmax':0.605,'opt_mode':'sgd', 'params':[0.098,0.0366,0.91],'dsb_tts':57766}
    if instance == 'G34':
        params_dic = {'N_batch':260 ,'N_step': 12000, 'N_rep':1000, "C_grad": 2.33, 'seed': 22, 'lr':2.638, 
                        'Tmin':6.24E-06, 'Tmax':0.605,'opt_mode':'sgd', 'params':[0.1182,0.0384,0.8967],'dsb_tts':2057}
    if instance == 'G35':
        params_dic = {'N_batch':20 ,'N_step': 15000, 'N_rep':10000, "C_grad": 1, 'seed': 83, 'lr':0.023, 
                        'Tmin':0.0001, 'Tmax':0.9,'opt_mode':'rmsprop', 'params':[0.9999,0.016,0.92],'dsb_tts':8319000}
    if instance == 'G36':
        params_dic = {'N_batch':25 ,'N_step': 12000, 'N_rep':10000, "C_grad": 1, 'seed': 14, 'lr':0.1, 
                        'Tmin':1.00E-03, 'Tmax':1,'opt_mode':'rmsprop', 'params':[0.999,0.0025,0.89],'dsb_tts':62646570}
    if instance == 'G37':
        params_dic = {'N_batch':20 ,'N_step': 10000, 'N_rep':10000, "C_grad": 1, 'seed': 24, 'lr':0.03, 
                        'Tmin':1.00E-04, 'Tmax':0.9,'opt_mode':'rmsprop', 'params':[0.999,0.02,0.92],'dsb_tts':27343457}
    if instance == 'G38':
        params_dic = {'N_batch': 260 ,'N_step': 7000, 'N_rep':1000, "C_grad": 1, 'seed': 52, 'lr':0.3, 
                        'Tmin': 8.00E-04, 'Tmax':0.4,'opt_mode':'rmsprop', 'params':[0.9999,0.0113,0.8595],'dsb_tts':98519}
    if instance == 'G39':
        params_dic = {'N_batch': 200, 'N_step': 7000, 'N_rep':1000, "C_grad": 1, 'seed': 49, 'lr':0.064, 
                        'Tmin': 1.50E-04, 'Tmax':0.76,'opt_mode':'rmsprop', 'params':[0.9999,0.0264,0.9081],'dsb_tts':56013}
    if instance == 'G40':
        params_dic = {'N_batch': 150, 'N_step': 10000, 'N_rep':1000, "C_grad": 1, 'seed': 4, 'lr':0.0525, 
                        'Tmin': 1.10E-04, 'Tmax':0.95,'opt_mode':'rmsprop', 'params':[0.9999,0.029,0.9082],'dsb_tts':24131}
    if instance == 'G41':
        params_dic = {'N_batch': 200, 'N_step': 12000, 'N_rep':1000, "C_grad": 4.61, 'seed': 86, 'lr':1.345, 
                        'Tmin': 1.32E-05, 'Tmax':0.655,'opt_mode':'sgd', 'params':[0.0725,0.0092,0.897],'dsb_tts':10585}
    if instance == 'G42':
        params_dic = {'N_batch': 10, 'N_step': 8000, 'N_rep':10000, "C_grad": 1, 'seed': 8, 'lr':0.096, 
                        'Tmin': 0.0001, 'Tmax':1,'opt_mode':'rmsprop', 'params':[0.9999,0.024,0.73275],'dsb_tts':550000}
    if instance == 'G43':
        params_dic = {'N_batch': 30, 'N_step': 1000, 'N_rep':1000, "C_grad": 1, 'seed': 91, 'lr':6.29, 
                        'Tmin': 0.0006, 'Tmax':0.65,'opt_mode':'sgd', 'params':[0.077,0.0285,0.7515],'dsb_tts':5.86}
    if instance == 'G44':
        params_dic = {'N_batch': 30, 'N_step': 1000, 'N_rep':1000, "C_grad": 1.2, 'seed': 22, 'lr':5.8, 
                        'Tmin': 0.0007, 'Tmax':0.65,'opt_mode':'sgd', 'params':[0.097,0.026,0.7554],'dsb_tts':6.5}
    
    if instance == 'G45':
        params_dic = {'N_batch': 70, 'N_step': 3000, 'N_rep':1000, "C_grad": 1.36, 'seed': 13, 'lr':6.1, 
                        'Tmin': 8.40E-04, 'Tmax':0.63,'opt_mode':'sgd', 'params':[0.129,0.01,0.755],'dsb_tts':43.4}
        
    if instance == 'G46':
        params_dic = {'N_batch': 120, 'N_step': 2000, 'N_rep':1000, "C_grad": 2.07, 'seed': 14, 'lr':1.54, 
                        'Tmin': 8.11E-06, 'Tmax':0.504,'opt_mode':'sgd', 'params':[0.156,0.0295,0.8965],'dsb_tts':16}
    
    if instance == 'G47':
        params_dic = {'N_batch': 70, 'N_step': 3000, 'N_rep':1000, "C_grad": 1, 'seed': 2, 'lr':7.5, 
                        'Tmin': 5.40E-04, 'Tmax': 0.58,'opt_mode':'sgd', 'params':[0.13,0.026,0.76],'dsb_tts':44.8}
    
    if instance == 'G48':
        params_dic = {'N_batch': 3, 'N_step': 180, 'N_rep':1000, "C_grad": 1, 'seed': 37, 'lr': 5.5, 
                        'Tmin': 0.001, 'Tmax': 1.34,'opt_mode':'sgd', 'params':[0.08,0.032,0.737],'dsb_tts':0.824}
        
    if instance == 'G49':
        params_dic = {'N_batch': 6, 'N_step': 200, 'N_rep':1000, "C_grad": 1, 'seed': 239, 'lr': 6.415, 
                        'Tmin': 5.90E-04, 'Tmax': 1.77,'opt_mode':'sgd', 'params':[0.42,0.073,0.572],'dsb_tts':0.784}
    
    if instance == 'G50':
        params_dic = {'N_batch': 10, 'N_step': 200, 'N_rep':1000, "C_grad": 0.833, 'seed': 88, 'lr': 0.436, 
                        'Tmin': 3.54E-05, 'Tmax': 22.94,'opt_mode':'sgd', 'params':[0.0617,0.0503,0.3335],'dsb_tts': 2.63}
    
    if instance == 'G51':
        params_dic = {'N_batch': 200, 'N_step': 7000, 'N_rep':1000, "C_grad": 1, 'seed': 82, 'lr': 1.345, 
                        'Tmin': 6.5e-06, 'Tmax': 1.48,'opt_mode':'sgd', 'params':[0.283,0.029,0.863],'dsb_tts': 12209}
        
    if instance == 'G52':
        params_dic = {'N_batch': 250, 'N_step': 10000, 'N_rep':1000, "C_grad": 2.4, 'seed': 57, 'lr': 2.9, 
                        'Tmin': 4.20e-06, 'Tmax': 0.604,'opt_mode':'sgd', 'params':[0.19,0.027,0.81],'dsb_tts': 6937}
    
    if instance == 'G53':
        params_dic = {'N_batch': 250, 'N_step': 10000, 'N_rep':1000, "C_grad": 10.8, 'seed': 95, 'lr': 6, 
                        'Tmin': 3.5e-06, 'Tmax': 0.27,'opt_mode':'sgd', 'params':[0.35,0.015,0.79],'dsb_tts': 93899}
        
    if instance == 'G54':
        params_dic = {'N_batch': 20, 'N_step': 10000, 'N_rep':10000, "C_grad": 6, 'seed': 36, 'lr': 1.27, 
                        'Tmin': 1e-05, 'Tmax': 0.63,'opt_mode':'sgd', 'params':[0.11,0.018,0.71],'dsb_tts': 2307235}
        
    return params_dic
    
if __name__ == '__main__':
    dev = 'cuda'  
    instance = 'G53'
    main(instance,dev)
    