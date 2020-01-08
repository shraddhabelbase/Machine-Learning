#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 17:25:54 2019

@author: shraddhabelbase
"""

import numpy as np
import matplotlib.pyplot as plt


class DecisionTree:
    def __init__(self, X, y, max_depth=2, depth=0,min_leaf_size=2,classes=None):
        self.X = np.array(X)
        self.y = np.array(y)
        self.n = len(y)
        self.depth = depth
        self.class_counts = []
        if classes is None:
            self.classes = np.unique(self.y)
        else:
            self.classes = classes
  
        for label in self.classes:
            self.class_counts.append(np.sum(label==self.y))
            
        self.class_counts = np.array(self.class_counts)
        self.prediction = self.classes[np.argmax(self.class_counts)] 
        class_ratios = self.class_counts / self.n
        self.gini = 1 - np.sum(class_ratios **2)
        
        
        if (depth == max_depth) or ( self.gini == 0):
            self.axis = None
            self.t = None
            self.left = None
            self.right = None
            return
        
        self.axis = 0
        self.t = 0
        best_gini = 2
        X_val = np.array(self.X)

        for item in range(X_val.shape[1]): 
            col_values = X_val[:,item].copy()
            col_values = np.sort(col_values) 
                
            for row in range (len(col_values)):
                sel = np.array(X_val[:,item] <= col_values[row])
                _,l_counts = np.unique(self.y[sel], return_counts=True) 
                _,r_counts = np.unique(self.y[~sel], return_counts=True)
            
                n_left = np.sum(sel)
                n_right = np.sum(~sel)
                if n_left == 0 :
                    left_gini = 1
                else:
                    left_gini = 1-np.sum((l_counts/n_left)**2)
                    
                if n_right == 0:
                    right_gini =1
                else:
                    right_gini = 1-np.sum((r_counts/n_right)**2)
                    
                gini = (n_left * left_gini + n_right * right_gini) / (n_left + n_right)

                if (gini <= best_gini):
                    self.axis = item
                    if((row +1)== len(col_values)):
                        self.t = col_values[row]
                    else:
                        self.t = (col_values[row]+col_values[row+1])/2
                    best_gini = gini
                    
        
        if (best_gini == 2) :
                  self.left = None
                  self.right = None
                  self.axis = None
                  self.t = None
                  return
        
        sel = X_val[:,self.axis]<=self.t
        self.left = DecisionTree(X[sel,:], y[sel], max_depth, depth+1, min_leaf_size,self.classes)
        self.right = DecisionTree(X[~sel,:], y[~sel], max_depth, depth+1, min_leaf_size,self.classes)
        
                
    def classify_row(self,row):
        row = np.array(row)
        
        if self.left == None or self.right == None:
            return self.prediction 
    
        if row[self.axis] <= self.t: 
            return self.left.classify_row(row)
        else:
            return self.right.classify_row(row)   

    def predict(self,X):
        X = np.array(X)
        predictions = []
        
        for i in range(X.shape[0]):
            row = X[i,:]
            predictions.append(self.classify_row(row))
            
        predictions = np.array(predictions)
        return predictions            
         
    def score(self,X,y):
        X_value = np.array(X)
        y_label = np.array(y)
        number_y = len(y_label)
        y_predicted = self.predict(X_value)     
        accuracy = (np.sum(y_label==y_predicted))/(number_y)      
        return accuracy
          
    def print_tree(self):
        msg = '  ' * self.depth + '* Size = ' + str(self.n)+' '+str(self.class_counts)
        msg += ', Gini: ' + str(round(self.gini,2))           
        if(self.left != None):
            msg += ', Axis: ' + str(self.axis)
            msg += ', Cut: ' + str(round(self.t,2))
        else:
            msg += ', Predicted Class: ' + str(self.prediction )
                
        
        print(msg)
        
        if self.left != None:
            self.left.print_tree()
            self.right.print_tree()
 #############################TEST CODE !####################################           
from sklearn.datasets import make_blobs


X1, y1 = make_blobs(n_samples=250, n_features=2, centers=4,
                    cluster_std=3, random_state=1)

X1_train, X1_val = X1[:200], X1[200:]
y1_train, y1_val = y1[:200], y1[200:]

tree_01 = DecisionTree(X1_train, y1_train, max_depth=3, min_leaf_size=2)

#plot_regions(tree_01, X1_train, y1_train)

tree_01.print_tree()

print()

print('Training Accuracy:  ', tree_01.score(X1_train, y1_train))
print('Validation Accuracy:', tree_01.score(X1_val, y1_val))