import concurrent.futures
import numpy as np
import os
import pandas as pd
import pickle
from collections import deque
from functools import partial
from mpmath import iv
from multiprocessing import Manager
from scipy import optimize
from time import time
from helper import leaf_contribution
h = 3
seed = 0
n_sample = 1000

def generate_data(seed):
    # Randomly sample 10-dimensional vectors
    np.random.seed(seed)
    z = np.random.exponential(1, 5 * n_sample).reshape(n_sample, 5)
    x = np.concatenate((np.array([z[:,0]]).T, np.array([z[:,0] + z[:,1]]).T, np.array([z[:,0] + z[:,1] + z[:,2]]).T, np.array([z[:,0] + z[:,1] + z[:,2] + z[:,3]]).T, np.array([z[:,0] + z[:,1] + z[:,2] + z[:,3] + z[:,4]]).T), axis = 1)
    u = np.random.rand(n_sample, 5)
    v = np.concatenate((x,u), axis = 1)

    # Calculate all six possible paths depending on the sampled vectors
    L2_01234 = np.linalg.norm([v[:,1]-v[:,0], h*(v[:,6]-v[:,5])],axis=0) + np.linalg.norm([v[:,2]-v[:,1], h*(v[:,7]-v[:,6])],axis=0) + np.linalg.norm([(v[:,3]-v[:,2]), h*(v[:,8]-v[:,7])],axis=0) + np.linalg.norm([(v[:,4]-v[:,3]), h*(v[:,9]-v[:,8])],axis=0)
    L2_01324 = np.linalg.norm([v[:,1]-v[:,0], h*(v[:,6]-v[:,5])],axis=0) + np.linalg.norm([v[:,3]-v[:,1], h*(v[:,8]-v[:,6])],axis=0) + np.linalg.norm([(v[:,2]-v[:,3]), h*(v[:,7]-v[:,8])],axis=0) + np.linalg.norm([(v[:,4]-v[:,2]), h*(v[:,9]-v[:,7])],axis=0)
    L2_02134 = np.linalg.norm([v[:,2]-v[:,0], h*(v[:,7]-v[:,5])],axis=0) + np.linalg.norm([v[:,1]-v[:,2], h*(v[:,6]-v[:,7])],axis=0) + np.linalg.norm([(v[:,3]-v[:,1]), h*(v[:,8]-v[:,6])],axis=0) + np.linalg.norm([(v[:,4]-v[:,3]), h*(v[:,9]-v[:,8])],axis=0)
    L2_02314 = np.linalg.norm([v[:,2]-v[:,0], h*(v[:,7]-v[:,5])],axis=0) + np.linalg.norm([v[:,3]-v[:,2], h*(v[:,8]-v[:,7])],axis=0) + np.linalg.norm([(v[:,1]-v[:,3]), h*(v[:,6]-v[:,8])],axis=0) + np.linalg.norm([(v[:,4]-v[:,1]), h*(v[:,9]-v[:,6])],axis=0)
    L2_03124 = np.linalg.norm([v[:,3]-v[:,0], h*(v[:,8]-v[:,5])],axis=0) + np.linalg.norm([v[:,1]-v[:,3], h*(v[:,6]-v[:,8])],axis=0) + np.linalg.norm([(v[:,2]-v[:,1]), h*(v[:,7]-v[:,6])],axis=0) + np.linalg.norm([(v[:,4]-v[:,2]), h*(v[:,9]-v[:,7])],axis=0)
    L2_03214 = np.linalg.norm([v[:,3]-v[:,0], h*(v[:,8]-v[:,5])],axis=0) + np.linalg.norm([v[:,2]-v[:,3], h*(v[:,7]-v[:,8])],axis=0) + np.linalg.norm([(v[:,1]-v[:,2]), h*(v[:,6]-v[:,7])],axis=0) + np.linalg.norm([(v[:,4]-v[:,1]), h*(v[:,9]-v[:,6])],axis=0)
    matrix = np.concatenate([np.array([L2_01234]),np.array([L2_01324]),np.array([L2_02134]),np.array([L2_02314]),np.array([L2_03124]),np.array([L2_03214])],axis=0)
    L2_min = matrix.min(axis=0); L2_argmin = matrix.argmin(axis=0)
    matrix = np.concatenate([v.T,matrix,np.array([L2_min, L2_argmin]),np.array([np.divide(L2_min,L2_01234)])],axis=0).T

    # Data
    X = matrix[:, :10]; Y = matrix[:, -2]
    global data
    data = np.concatenate([X, np.array([Y]).T], axis = 1)
    data = pd.DataFrame(data, columns = ['x0','x1','x2','x3','x4','y0','y1','y2','y3','y4','label'])

# Auxiliary functions

def calc_cross_entropy(data):
    classes, class_counts = np.unique(data['label'], return_counts = True)
    probs = class_counts / len(data)
    cross_entropy = - np.sum(probs * np.log(probs))
    return cross_entropy

def calc_info_gain(threshold, feature, data):
    data_entropy = calc_cross_entropy(data)
    branch1 = data[data[feature] >= threshold]; branch1_entropy = calc_cross_entropy(branch1)
    branch2 = data[data[feature] < threshold]; branch2_entropy = calc_cross_entropy(branch2)
    info_gain = data_entropy - len(branch1)/len(data)*branch1_entropy - len(branch2)/len(data)*branch2_entropy
    if not (branch1_entropy == branch2_entropy == 0):
        return info_gain # not leaf node
    else:
        return info_gain - (min(branch1[feature]) - threshold) * (max(branch2[feature]) - threshold) # leaf node, max margin

def calc_threshold(data):
    info_gain = 0
    for feature in set(['x0','x1','x2','x3','x4']):
        x_max = max(data[feature])
        f_thres = lambda thres: - calc_info_gain(thres, feature = feature, data = data)
        opt = optimize.minimize_scalar(f_thres, bounds = [0,x_max], method = 'bounded')
        threshold = opt.x; objective = - opt.fun
        if objective > info_gain:
            info_gain = objective
            partition = (feature, threshold)
    for feature in set(['y0','y1','y2','y3','y4']):
        f_thres = lambda thres: - calc_info_gain(thres, feature = feature, data = data)
        opt = optimize.minimize_scalar(f_thres, bounds = [0,1], method = 'bounded')
        threshold = opt.x; objective = - opt.fun
        if objective > info_gain:
            info_gain = objective
            partition = (feature, threshold)
    return partition

class Node:
    def __init__(self, feature, threshold, ancestor):
        self.left = None
        self.right = None
        self.label = None
        self.feature = feature
        self.threshold = threshold
        self.ancestor = list(ancestor)
    
    def insert(self, leftchild, rightchild):
        self.left = leftchild
        self.right = rightchild

class Tree:
    def __init__(self):
        self.ancestor = deque()
    
    def create_decision_tree(self, data, leaf_nodes):
        unique_label = np.unique(data['label'])
        if len(unique_label) <= 1:
            leaf = Node(np.nan, np.nan, self.ancestor)
            leaf.label = unique_label[0]
            leaf_nodes.append(leaf)
            return leaf
        else:
            feature, threshold = calc_threshold(data)
            threshold = threshold.item()
            parent = Node(feature, threshold, self.ancestor)
            bool = data[feature] < threshold
            self.ancestor.append((feature, threshold, 'L'))
            leftchild = self.create_decision_tree(data[bool], leaf_nodes)
            self.ancestor.pop()
            self.ancestor.append((feature, threshold, 'R'))
            rightchild = self.create_decision_tree(data[np.logical_not(bool)], leaf_nodes)
            self.ancestor.pop()
            parent.insert(leftchild, rightchild)
            return parent

def worker(leaf):
    saving = leaf_contribution(leaf)
    if isinstance(saving, list):
        saving = '|'.join([str(x) for x in saving])
    else:
        saving = str(saving)
    return saving

if __name__ == '__main__':
    generate_data(seed)
    start_tree = time()
    leaf_nodes = Manager().list()
    tree = Tree()
    tree.create_decision_tree(data, leaf_nodes) # Comment this line and uncomment the two lines below to save tree
    #with open('tree_' + str(n_sample), 'wb') as file:
    #    pickle.dump(tree.create_decision_tree(data, leaf_nodes), file)
    end_tree = time()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(worker, leaf_nodes))
    parsed_results = []
    for result in results:
        parsed_results.append(iv.mpf(result))
    total_improvement = sum(parsed_results)
    end_node = time()
    print(f"Time to build tree (sec): {end_tree - start_tree:.2f}")
    print(f"Time to calculate improvement (sec): {end_node - end_tree:.2f}")
    print(f"Total upper bound improvement as interval arithmetic: {total_improvement}")
