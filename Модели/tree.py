# -*- coding: utf-8 -*-
"""
Created on Fri May 29 13:17:13 2020

@author: stife
"""

class tree:
    
    def __init__(self, key=None, data=None, l=None, r=None):
        self.key = key
        self.data = data
        self.left = l
        self.right = r
        
    def __str__(self):
        return "key = " + str(self.key) + " data = " + str(self.data)
    
    def is_leaf(self):
        return self.left == None and self.right == None
    
def print_tree(root, height=0):
    if not root is None:
        print_tree(root.right, height + 1)
        print(" " * 2 * height, root)
        print_tree(root.left, height + 1)

