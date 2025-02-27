#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 15:22:05 2024

@author: yizhou
"""

import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import os, re, shutil
from ExchangeFoam import readStatefromFoam, act2Foam, coeffsfromFoam
from LSTMenv import predictReward, evaluate_model

class openFoamEnv(gym.Env):
    
    def __init__(self,rootPath, actLowLimit, actHighLimit, 
                 obsLowLimit, obsHighLimit, timeLimit, deltaTime,
                 referenceP,alpha,scaleFactor, processor, n_obs, seq_len):
        super().__init__()
        
        self.rootPath = rootPath
        self.timeLimit = timeLimit
        self.episodeTime = None
        self.deltaTime = deltaTime
        self.referenceP = referenceP
        self.alpha = alpha
        self.scaleFactor = scaleFactor
        self.processor = processor
        
        self.action_space = spaces.Box(low=actLowLimit, high=actHighLimit, dtype=np.float32)
        self.observation_space = spaces.Box(low=obsLowLimit, high=obsHighLimit,shape=(n_obs,), dtype=np.float32)
        self.state = None
        self.numEpisode = 0
        self.Model = None
        
        self.seq_len = seq_len
        
    def reset(self, seed=None):
        
        for fileName in os.listdir(self.rootPath):
            if re.search(r'^\d+\.?\d*', fileName):
                if fileName != '0':
                    shutil.rmtree('/'.join([self.rootPath, fileName]))
            elif fileName == 'postProcessing':
                shutil.rmtree('/'.join([self.rootPath, fileName]))   
            # elif fileName == 'log':
            #     os.remove('/'.join([self.rootPath, fileName]))    
        
        self.episodeTime = 0
        self.lastAction = np.zeros(self.action_space.shape[-1])
        
        self.state = readStatefromFoam(self.episodeTime, self.rootPath, self.referenceP)
        self.state_save = []
        self.action_save = []
        self.reward_save = []
        
        
        self.numEpisode = self.numEpisode + 1
        print(f'numEpisode: {self.numEpisode}')
        
        return self.state, {}
    
    def step(self, action):
        
        
        action = action+1
        action = action*self.scaleFactor
        np.clip(action,100,self.scaleFactor*2,out=action)
        
        
      
        with  open ('/'.join([self.rootPath,'logLSTM']),'a') as fopenLSTM:
        
            if self.numEpisode <=self.seq_len or (self.numEpisode>self.seq_len and self.numEpisode%10==0):  #CFD environment
            
                with open ('/'.join([self.rootPath,'log']),'a') as fopen:  
                            
                    
                     
                    fopen.write(f'net: {action}\n' ) 
                    fopenLSTM.write(f'net: {action}\n' ) 
                    action = self.lastAction+ self.alpha*(action-self.lastAction)           
                    fopen.write(f'last: {self.lastAction}\n')
                    fopen.write(f'out: {action}\n')
                    fopenLSTM.write(f'last: {self.lastAction}\n')
                    fopenLSTM.write(f'out: {action}\n')
                    self.action_save.append(action[0])
                    self.state = act2Foam(action, self.episodeTime, self.rootPath, self.deltaTime, self.processor, self.referenceP) 
                    with open ('/'.join([self.rootPath,'logState']),'a') as fopenState:
                       np.savetxt(fopenState, self.state)
                    
                    reward = coeffsfromFoam(self.episodeTime+self.deltaTime, self.rootPath, self.referenceP, action)
                    fopen.write(f'reward = {reward}\n')
                    fopenLSTM.write(f'reward = {reward}\n')
                self.reward_save.append(reward)
                self.state_save.append(self.state)   
                if round(self.episodeTime,5)== round(self.timeLimit-self.deltaTime,5):
                    if self.numEpisode>=self.seq_len and self.numEpisode%10==0:
                        if self.numEpisode>self.seq_len:
                            reward_predicted = evaluate_model(self.Model,self.seq_len, np.array(self.action_save))
                            MSEloss = np.mean((
                            np.concatenate((np.expand_dims(np.array(self.reward_save),axis=1),
                            np.array(self.state_save)),axis=1)
                            -reward_predicted)**2)
                            print('predicted reward loss:', MSEloss)
                        
                        
                        self.Model = predictReward(self.rootPath,  self.seq_len)
                        
                            
                    
            else:   # LSTM environment
                
               
                fopenLSTM.write(f'net: {action}\n' )         
                action = self.lastAction+ self.alpha*(action-self.lastAction)           
                fopenLSTM.write(f'last: {self.lastAction}\n')
                fopenLSTM.write(f'out: {action}\n')
                self.action_save.append(action[0])
                
                x_test = np.pad(np.array(self.action_save),(0,50-len(self.action_save)), 'constant')
                y_test = evaluate_model(self.Model,self.seq_len,x_test)
                reward = y_test[len(self.action_save)-1,0]
                fopenLSTM.write(f'reward = {reward}\n')
            
              
                self.state = y_test[len(self.action_save)-1,1:]
        
        
        self.episodeTime = self.episodeTime + self.deltaTime
        done = self.episodeTime >= self.timeLimit                                                
        self.lastAction = action
                   
            
        return self.state, reward, done, False, {}
    
    
    def render(self, mode='human'):
        pass
    
    def close(self):
        pass
        
