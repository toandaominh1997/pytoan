import numpy as np 


class Adaboost(object):
    def __init__(self, n_clf):
        self.n_clf = n_clf
    
    def fit(self, X, y):

        for _ in range(self.n_clf):
            
