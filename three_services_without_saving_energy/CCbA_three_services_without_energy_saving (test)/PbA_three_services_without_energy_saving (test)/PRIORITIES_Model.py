
import torch 
import numpy as np
import pandas as pd

from collections import deque

import random

from configurations import *


import copy

import math


# %%

random.seed(0)

torch.autograd.set_detect_anomaly(True)

device=torch.device('cpu')
# device=torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# %%

# %%

class PriorityModel(torch.nn.Module):
    def __init__(self,num_inputs,num_outputs,hidden_layers):
        super(PriorityModel,self).__init__()
        
        self.num_inputs=num_inputs
        
        self.num_output_features=num_outputs
        
        self.hidden_layers=hidden_layers
        
        self.lstm1=torch.nn.LSTMCell(self.num_inputs,self.hidden_layers)
        self.lstm2=torch.nn.LSTMCell(self.hidden_layers,self.hidden_layers)
        self.lstm3=torch.nn.LSTMCell(self.hidden_layers,self.hidden_layers)
        self.linear=torch.nn.Linear(self.hidden_layers,self.num_output_features)
            
    def forward(self,y):
       
        
        if len(y.shape)!=3:
            y=torch.from_numpy(np.expand_dims(y, axis=0))
            
    
        n_samples =y.size(0)
        
        h_t, c_t = self.lstm1(y[0])
        h_t2, c_t2 = self.lstm2(h_t)
        h_t3, c_t3 = self.lstm2(h_t2)
        
        output = self.linear(h_t3)
       

        return output    