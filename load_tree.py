import pickle
from time import time
n_sample = 1000

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

start_tree = time()
with open('Example trees/tree_' + str(n_sample), 'rb') as file:
    loaded_tree = pickle.load(file)
end_tree = time()
print(f"Time to load tree (sec): {end_tree - start_tree:.2f}")

# Tree
print(loaded_tree)
print(loaded_tree.feature, loaded_tree.threshold)
print(loaded_tree.ancestor)

# Left child
print(loaded_tree.left)
print(loaded_tree.left.feature, loaded_tree.left.threshold)
print(loaded_tree.left.ancestor)

# Right child
print(loaded_tree.right)
print(loaded_tree.right.feature, loaded_tree.right.threshold)
print(loaded_tree.right.ancestor)

# Left-left child
print(loaded_tree.left.left)
print(loaded_tree.left.left.feature, loaded_tree.left.left.threshold)
print(loaded_tree.left.left.ancestor)

# Left-right child
print(loaded_tree.left.right)
print(loaded_tree.left.right.feature, loaded_tree.left.right.threshold)
print(loaded_tree.left.right.ancestor)

# Right-left child
print(loaded_tree.right.left)
print(loaded_tree.right.left.feature, loaded_tree.right.left.threshold)
print(loaded_tree.right.left.ancestor)

# Right-right child
print(loaded_tree.right.right)
print(loaded_tree.right.right.feature, loaded_tree.right.right.threshold)
print(loaded_tree.right.right.ancestor)
