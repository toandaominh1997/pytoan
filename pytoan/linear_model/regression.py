import numpy as np 
import torch 
import torch.nn as nn 

import torch.nn.functional as F 


class Regression(nn.Module):
    def __init__(self):
        super(Regression, self).__init__() 
    def init_weights(self, X):
        batch_size, in_features = X.size()
        self.linear = nn.Linear(in_features, batch_size,bias=True)


class LinearRegression(Regression):
    def forward(self, X):
        self.init_weights(X)
        y_pred = self.linear(X)
        return y_pred

class LassoRegression(Regression):
    def forward(self, X):


if __name__=='__main__':
    X = torch.randn(4, 128)
    model = LinearRegression()
    y_pred = model(X)
    print('predict: ', y_pred.size())
