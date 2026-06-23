
import torch
import networkx as nx

def load_file_sparse(test_instance):
    G = nx.Graph()
    # load Gset instance file
    path = 'Gset/'+ test_instance  # file path
    with open(path,"r") as f:
        l = f.readline()
        N, m = [int(x) for x in l.split(" ") if x!="\n"] # n: nodes, m:edges
        # W = np.zeros([N,N])
        G.add_nodes_from(range(N))
        sum_w = 0
        for k in range(m):
            l = f.readline()
            l = l.replace("\n","")
            i,j,w = [int(x) for x in l.split(" ")]
            G.add_edge(i-1,j-1, weight = -w)
            sum_w += w
        sum_w = 0.5 * sum_w
        # J = -W.copy()
    crow_indices = [0]
    col_indices = []
    values = []
    for node in range(N):
        neighbors = list(G.neighbors(node))
        if len(neighbors) == 0:
            crow_indices.append(crow_indices[-1])
        else:
            for neighbor in neighbors:
                col_indices.append(neighbor)
                values.append(G.edges[node, neighbor]['weight'])
            crow_indices.append(len(values))


    crow_indices = torch.tensor(crow_indices)
    col_indices = torch.tensor(col_indices)
    values = torch.tensor(values)
    J = torch.sparse_csr_tensor(crow_indices, col_indices, values, dtype=torch.float32)

    return G, J , sum_w

G,J,sum_w = load_file_sparse('G1')
J = J.to('cuda:2')
print(J.device)