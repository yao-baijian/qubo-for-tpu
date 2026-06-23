import numpy as np
import torch
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
import warnings
warnings.filterwarnings('ignore')


def load_matrix(path:'str',numer_package:'str',store_format:'str',return_n_m:'bool'=False) -> 'float':
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
        return J
    else:
        return J, N, edges


if __name__ == '__main__':
    path = './balanced_min_cut/Graphs/bcsstk33.txt'
    J1 = load_matrix(path,'scipy','csr')
    print(J1)
    J2 = load_matrix(path,'torch','csr')
    print(torch.allclose(torch.from_numpy(J1.todense()).to(torch.float32),
                                          J2.to_dense().to(torch.float32)))
