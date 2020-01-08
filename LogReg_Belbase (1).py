#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 09:44:57 2019

@author: shraddhabelbase
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class LogisticRegression:
    
    def __init__(self, X, y):
        self.X = np.array(X) #store the training features
        self.y = np.array(y) #contain the training labels
        self.N = len(y) #the number of training observations
        self.classes = np.unique(self.y)
            
        def NLL(coefficients): #store coefficients
            ones = np.ones((self.N, 1))
            X_ = np.hstack((ones, self.X))
            z = np.dot(X_,coefficients)
            z = z.reshape(-1,)
            y_predicted = coefficients[0] + np.sum(coefficients[1:] *X, axis=1)
            e = 1 / (1 + np.exp((-1) * y_predicted))
            ei = np.where(self.y == self.classes[1], e, 1 - e)
            loglik = np.sum(np.log(ei))
            return loglik * (-1)
        
        np.seterr(all = "ignore")
        beta_guess = np.zeros(X.shape[1] + 1)
        min_results = minimize(NLL, beta_guess)
        self.coefficients = min_results.x
        np.seterr(all = "warn")
        
        self.loglik = round(NLL(self.coefficients) * (1),4)
        self.y_predicted = self.predict(X) #array
        self.accuracy = self.score(self.X, self.y)
        
        
    def predict_proba(self, X):
        X = np.array(X) # X into array
        y_predicted = self.coefficients[0] + np.sum(self.coefficients[1:] *X, axis=1)
        return 1 / (1 + np.exp((-1) * y_predicted))
       
    def predict(self, X, t = 0.5): #float t default
         X = np.array(X)
         p = self.predict_proba(X)
         labels = np.where(p < t, self.classes[0], self.classes[1])
         return labels
            
    def summary(self):
        print('+----------------------------+')
        print('| Logistic Regression Summary|')
        print('+----------------------------+')
        print('Number of training observations: ' + str(self.N))
        print('Coefficient Estimates:' + str(self.coefficients))
        print('Negative Log-Likelihood: ' + str(self.loglik))
        print('Accuracy: ' + str(self.accuracy) + "\n")
       
             
    def score(self, X, y, t = 0.5):        
        X = np.array(X)
        y = np.array(y)
        y_predicted = self.predict(X)
        score = np.sum(y_predicted == y) / len(y)
        
        return score
    
    def confusion_matrix(self, X, y, t = 0.5):
        X = np.array(X)
        y = np.array(y)
        
        y_predicted = self.predict(self.X, t)
        TP = np.sum((self.y == self.classes[0]) & (y_predicted == self.classes[0]))
        FP = np.sum((self.y == self.classes[0]) & (y_predicted == self.classes[1]))
        TN = np.sum((self.y == self.classes[1]) & (y_predicted == self.classes[0]))
        FN = np.sum((self.y == self.classes[1]) & (y_predicted == self.classes[1]))    
        cm = np.array([[TP, FP],[TN, FN]])
        print(cm)
        
    def precision_recall(self,X,y,t =0.5):
        X = np.array(X)
        y = np.array(y)
        y_predicted = self.predict(self.X, t)
        TP = np.sum((self.y == self.classes[0]) & (y_predicted == self.classes[0]))
        FP = np.sum((self.y == self.classes[0]) & (y_predicted == self.classes[1]))
        TN = np.sum((self.y == self.classes[1]) & (y_predicted == self.classes[0]))
        FN = np.sum((self.y == self.classes[1]) & (y_predicted == self.classes[1])) 
        
        print('Class :' + str(self.classes[0]))
        per = round(TP/(TP+TN),4)
        print('  Precision = ' + str(per))
        rec = round(TP/(TP+FP),4)
        print('  Recall    = ' + str(rec))
        
        print('Class :' + str(self.classes[1]))
        per2 = round(FP/(FP+TN),4)
        print('  Precision = ' + str(per2))
        rec2 = round(FN/(FN+TN),4)
        print('  Recall    = ' + str(rec2))
        

#################################TEST CODE2 ###########################
np.set_printoptions(suppress=True, precision=4)
width = [6.4, 7.7, 6.7, 7.4, 6.5, 6.9, 7.8, 7.6, 6.2, 7.4, 7.7, 6.8]
height = [8.2, 7.5, 6.6, 8.8, 6.8, 6.8, 7.6, 8.8, 8.4, 7.3, 7.4, 7.2]
X = pd.DataFrame({'x1':width, 'x2':height})
y = ['Lemon', 'Orange', 'Orange', 'Lemon', 'Orange', 'Lemon', 'Orange', 'Lemon', 'Lemon', 'Orange', 'Lemon', 'Lemon']

model_02 = LogisticRegression(X,y)
model_02.summary()

model_02.confusion_matrix(X,y)

model_02.precision_recall(X, y)

X_test = pd.DataFrame({'x1':[7.4, 7.1, 6.4, 6.9, 5.8], 'x2':[7.2, 7.8, 6.8, 6.7, 6.4]})
y_test = ['Orange', 'Orange', 'Lemon', 'Orange', 'Lemon']

print("Test Set Performance:" +"\n")

print(model_02.predict_proba(X_test))
print(model_02.predict(X_test))
print(model_02.score(X_test,y_test))


