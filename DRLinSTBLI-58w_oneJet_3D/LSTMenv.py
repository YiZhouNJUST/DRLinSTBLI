#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 19:07:03 2024

@author: yizhou
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def getAction(rootPath):
    
    fopen = fopen = open('/'.join([rootPath, 'log']))
    p = fopen.readlines()
    fopen.close()
    pout = p[2::4]
    reward = p[3::4]
    
    
    pnumber=[]
    for item in pout:
        action = item.split()[1].replace('[','').replace(']','')
        pnumber.append(float(action))
    pnumber = np.array(pnumber)
    
    rnumber=[]
    for item in reward:
        r = item.split()[2].replace('[','').replace(']','')
        rnumber.append(float(r))
    rnumber = np.array(rnumber)
    
       
    return pnumber,rnumber

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        return output
    
def evaluate_model(model,seq_len,x_test):
    model.eval()
    with torch.no_grad():
        x_test = torch.tensor(x_test, dtype=torch.float32).view(-1, seq_len, 1)
        test_output = model(x_test)
        test_output = test_output.detach().numpy().squeeze()
        return test_output

def predictReward(rootPath,seq_len):
    action, reward= getAction(rootPath)
    reward = reward[:reward.shape[0]//seq_len*seq_len].reshape(-1,seq_len)
    action = action[:action.shape[0]//seq_len*seq_len].reshape(-1,seq_len)
    state = np.loadtxt(rootPath+'/logState')
    state = state[:state.shape[0]//(seq_len*168)*(seq_len*168)].reshape(-1,seq_len,168)
    x_train = action.copy()
    # y_train = state.copy()
    y_train = np.concatenate((np.expand_dims(reward, axis=2),state),axis=2)
    
    
    
   
    input_size = 1    
    hidden_size = 64 
    num_layers = 3  
    output_size = 169  
    num_epochs = 2000  
    learning_rate = 0.001
            
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    x_train = torch.tensor(x_train, dtype=torch.float32).view(-1, seq_len, input_size)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, seq_len, output_size)
   
   
    
    losses = []  
    # TestLosses = []
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()        
        output = model(x_train)        
        loss = criterion(output, y_train)        
        loss.backward()
        optimizer.step()        
        losses.append(loss.item())
        
    with open ('/'.join([rootPath,'logTrainLoss']),'a') as fopenLoss:
       np.savetxt(fopenLoss, losses)
    
    return  model
        
    
if __name__ == '__main__':
    rootPath = '/totalPBc'
    
    seq_len = 50
   
    model = predictReward(rootPath, seq_len)  
    
    
    
 

    


