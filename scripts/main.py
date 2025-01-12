import concurrent.futures
import numpy as np
import random
import os
from multiprocessing import Manager
from collections import deque
from mpmath import iv
from time import time
from helper import leaf_contribution
h = 3.25
seed = 0
n_sample = 10000

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
    
    X = matrix[:, :10]; Y = matrix[:, -2]
    global data
    data = np.concatenate([X, np.array([Y]).T], axis = 1)

def calc_cross_entropy(data):
    classes, class_counts = np.unique(data[:, -1], return_counts = True)
    probs = class_counts / len(data)
    cross_entropy = - np.sum(probs * np.log(probs))
    return cross_entropy

def calc_info_gain(threshold, feature, data):
    data_entropy = calc_cross_entropy(data)
    branch1 = data[data[:, feature] >= threshold, :]; branch1_entropy = calc_cross_entropy(branch1)
    branch2 = data[data[:, feature] < threshold, :]; branch2_entropy = calc_cross_entropy(branch2)
    info_gain = data_entropy - len(branch1)/len(data)*branch1_entropy - len(branch2)/len(data)*branch2_entropy
    return info_gain

def grad_desc(f, lb, ub, lr = 1e-2, iter = 100, tol = 1e-4):
    x, d = (ub + lb)/2, (ub - lb)*1e-4
    for i in range(iter):
        grad = (f(x + d) - f(x - d))/(2*d)
        new_x = x - lr * grad
        if new_x < lb or new_x > ub:
            return x
        elif abs(new_x - x) < tol:
            return new_x
        x = new_x
    return new_x

def calc_threshold(data):
    partition = None
    info_gain = 0
    order = deque([0,1,2,3,4,5,6,7,8,9])
    random.shuffle(order)
    for feature in order:
        LB = min(data[:, feature]); UB = max(data[:, feature])
        f_thres = lambda thres: - calc_info_gain(thres, feature = feature, data = data)
        threshold = grad_desc(f = f_thres, lb = LB, ub = UB)
        objective = - f_thres(threshold)
        if objective > info_gain:
            info_gain = objective
            partition = (feature, threshold)
    if partition is not None:
        return partition
    else:
        return (feature, threshold)

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
        unique_label = np.unique(data[:, -1])
        if len(unique_label) <= 1:
            leaf = Node(np.nan, np.nan, self.ancestor)
            leaf.label = unique_label[0]
            leaf_nodes.append(leaf)
            return leaf
        else:
            feature, threshold = calc_threshold(data)
            threshold = threshold.item()
            parent = Node(feature, threshold, self.ancestor)
            bool = data[:, feature] < threshold
            self.ancestor.append((feature, threshold, 'L'))
            leftchild = self.create_decision_tree(data[bool], leaf_nodes)
            self.ancestor.pop()
            self.ancestor.append((feature, threshold, 'R'))
            rightchild = self.create_decision_tree(data[np.logical_not(bool)], leaf_nodes)
            self.ancestor.pop()
            parent.insert(leftchild, rightchild)
            return parent

def worker(leaf):
    saving = str(leaf_contribution(leaf))
    return saving if saving != 0 else None

def chunkify(lst, chunk_size = 10000000):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]
        
def process_chunk(chunk, chunk_index):
    print(f"Processing chunk {chunk_index}")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        return list(executor.map(worker, chunk))

if __name__ == '__main__':
    generate_data(seed)
    start_tree = time()
    tree = Tree(); leaf_nodes = []
    tree.create_decision_tree(data, leaf_nodes)
    del data
    end_tree = time()
    print(f"Time to build tree (sec): {end_tree - start_tree:.2f}")
    
    results = deque(); parsed_results = deque()
    leaf_node_chunks = chunkify(leaf_nodes)
    for index, chunk in enumerate(leaf_node_chunks):
        chunk_result = process_chunk(chunk, index+1)
        results.extend(chunk_result)
    for result in results:
        parsed_results.append(iv.mpf(result))
    total_saving = sum(parsed_results)
    end_node = time()
    print(f"Time to calculate saving (sec): {end_node - end_tree:.2f}")
    print(f"Total saving in interval arithmetic: {total_saving}")
