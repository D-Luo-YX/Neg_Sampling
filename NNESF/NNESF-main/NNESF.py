# -*- coding: utf-8 -*-



import networkx as nx
import random
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
import multiprocessing as mp
import scipy.io as scio
from functools import partial
import warnings

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def training_graph_ideal(G,p,is_sampled=True,r=1):
    edges = list(G.edges())
    num_edges = len(edges)
    num_test_edges = round(p*num_edges)
    num_train_edges = num_edges-num_test_edges
    G2 = G.copy()
    del_egdes = random.sample(edges,num_test_edges)
    G2.remove_edges_from(del_egdes)
    train_pos_edges = list(G2.edges())
    test_pos_edges = del_egdes
    
    non_edges = list(nx.non_edges(G))
    ind = list(np.random.permutation(len(non_edges)))
    ind_train_non_edges = ind[:int(r*num_train_edges)]
    if is_sampled:
        ind_test_non_edges = ind[int(r*num_train_edges):int(r*num_train_edges)+int(num_test_edges)]
    else:
        ind_test_non_edges = ind[int(r*num_train_edges):]
    train_non_edges = [non_edges[i] for i in ind_train_non_edges]
    test_non_edges = [non_edges[i] for i in ind_test_non_edges]
    
    return G2, train_pos_edges,train_non_edges,test_pos_edges,test_non_edges


def training_graph_not_ideal(G, train_pos_edges,train_non_edges,test_pos_edges,test_non_edges, theta):
    max_replacement = min(len(train_non_edges) * theta, len(test_pos_edges))
    num_replacements = int(max_replacement)

    if num_replacements > 0:
        replacement_edges = random.sample(test_pos_edges, num_replacements)
        train_non_edges = train_non_edges[:-num_replacements] + replacement_edges

    return G, train_pos_edges, train_non_edges, test_pos_edges,test_non_edges


def subgraph2vec(ebunch,G2):
    x,y = ebunch
    nei_x = set(G2[x])
    nei_x.discard(y)
    sub_x = G2.subgraph(nei_x)
    nei_y = set(G2[y])
    nei_y.discard(x)
    sub_y = G2.subgraph(nei_y)
    cn_xy = set(nx.common_neighbors(G2, x, y))
    sub_cn = G2.subgraph(cn_xy)
    
    fea = np.zeros(6)
    fea[0] = sub_cn.number_of_edges()
    fea[1] = sub_cn.number_of_nodes()
    fea[2] = sub_x.number_of_edges()
    fea[3] = sub_x.number_of_nodes()
    fea[4] = sub_y.number_of_edges()
    fea[5] = sub_y.number_of_nodes()
    
    return fea


def graph2vector(G2, train_pos_edges,train_non_edges,test_pos_edges,test_non_edges):
    partial_work = partial(subgraph2vec,G2=G2)
    train_pos_features = pool.map(partial_work,train_pos_edges)
    train_non_features = pool.map(partial_work,train_non_edges)
    test_pos_features = pool.map(partial_work,test_pos_edges)
    test_non_features = pool.map(partial_work,test_non_edges)
    
    train_features = train_pos_features+train_non_features
    train_labels = [1]*len(train_pos_features)+[0]*len(train_non_features)
    test_features = test_pos_features+test_non_features
    test_labels = [1]*len(test_pos_features)+[0]*len(test_non_features)
    
    train_features = np.array(train_features)
    train_labels = np.array(train_labels)
    test_features = np.array(test_features)
    test_labels = np.array(test_labels)
    
    return train_features, train_labels, test_features, test_labels


def symmetry(train_features, train_labels, test_features, test_labels):
    train_features_xy = train_features.copy()
    train_features_yx = train_features.copy()
    train_features_yx[:,2] = train_features[:,4]
    train_features_yx[:,3] = train_features[:,5]
    train_features_yx[:,4] = train_features[:,2]
    train_features_yx[:,5] = train_features[:,3]
    train_features = np.vstack((train_features_xy,train_features_yx))
    train_labels = np.concatenate((train_labels,train_labels))
    
    test_features_xy = test_features.copy()
    test_features_yx = test_features.copy()
    test_features_yx[:,2] = test_features[:,4]
    test_features_yx[:,3] = test_features[:,5]
    test_features_yx[:,4] = test_features[:,2]
    test_features_yx[:,5] = test_features[:,3]
    
    return train_features, train_labels,test_features_xy,test_features_yx,test_labels


def cal_mean_std(auc):
    mean = np.sum(auc,1)/np.size(auc,1)
    std = np.zeros(np.size(auc,0))
    for i in range(np.size(auc,0)):
        std[i] = np.std(auc[i,:])
    mean = mean*100
    std = std*100
    return mean,std


def extra_and_pre(G2, train_pos_edges,train_non_edges,test_pos_edges,test_non_edges):
    train_features, train_labels, test_features, test_labels = graph2vector(
        G2, train_pos_edges,train_non_edges,test_pos_edges,test_non_edges)
    
    train_features, train_labels,test_features_xy,test_features_yx,test_labels = \
        symmetry(train_features, train_labels, test_features, test_labels)
    
    clf = MLPClassifier(hidden_layer_sizes=(32, 32, 16), alpha=1e-3,
                          batch_size=128, learning_rate_init=0.001,
                          max_iter=100)
    clf.fit(train_features, train_labels)
    
    y_pred_xy = clf.predict_proba(test_features_xy)[:,1]
    y_pred_yx = clf.predict_proba(test_features_yx)[:,1]
    y_pred = (y_pred_xy+y_pred_yx)/2
    res = roc_auc_score(test_labels, y_pred)
    
    return res

if __name__=='__main__':
    p = 0.1
    n_loops = 10
    random_seed = 1
    dataset = ['USAir.mat','NS.mat','PB.mat','Yeast.mat','Celegans.mat',
               'Power.mat','Router.mat','Ecoli.mat','facebook.mat','Wiki-Vote.mat']#
    # dataset = ['facebook.mat']
    # dataset = ['USAir.mat','PB.mat']
    auc_ideal = np.zeros((len(dataset),n_loops))
    auc_not_ideal = np.zeros((len(dataset),n_loops))
    pool = mp.Pool()#processes=5
    warnings.filterwarnings('ignore')
    print("")
    for i,data in enumerate(dataset):
        random_seed = 1
        print(f"{data}")
        A = scio.loadmat('dataset/'+data)['net'].A
        G = nx.from_numpy_matrix(A)
        
        for j in range(n_loops):
            set_random_seed(random_seed)
            G2, train_pos_edges,train_non_edges,test_pos_edges,test_non_edges = training_graph_ideal(G, p)
            set_random_seed(random_seed)
            auc_ideal[i,j] = extra_and_pre(G2, train_pos_edges,train_non_edges,test_pos_edges,test_non_edges)
            set_random_seed(random_seed)
            G2_ni, train_pos_edges_ni, train_non_edges_ni, test_pos_edges_ni, test_non_edges_ni = training_graph_not_ideal(G2, train_pos_edges,train_non_edges,test_pos_edges,test_non_edges,theta=0.001)
            set_random_seed(random_seed)
            auc_not_ideal[i, j] = extra_and_pre(G2_ni, train_pos_edges_ni, train_non_edges_ni, test_pos_edges_ni, test_non_edges_ni)
            random_seed+=1

    mean,std = cal_mean_std(auc_ideal)
    mean_ni, std_ni = cal_mean_std(auc_not_ideal)
    print(mean)
    print(std)
    print(mean_ni)
    print(std_ni)
    print(auc_ideal)
    print(auc_not_ideal)

    # Save results to .txt files
    with open("auc_ideal.txt", "w") as f_ideal:
        f_ideal.write("Dataset: {}\n".format(dataset))
        for i, data in enumerate(dataset):
            f_ideal.write(f"{data} AUC Ideal:\n")
            for j in range(n_loops):
                f_ideal.write(f"{auc_ideal[i, j]:.6f}\n")
            f_ideal.write("\n")

    with open("auc_not_ideal.txt", "w") as f_not_ideal:
        f_not_ideal.write("Dataset: {}\n".format(dataset))
        for i, data in enumerate(dataset):
            f_not_ideal.write(f"{data} AUC Not Ideal:\n")
            for j in range(n_loops):
                f_not_ideal.write(f"{auc_not_ideal[i, j]:.6f}\n")
            f_not_ideal.write("\n")