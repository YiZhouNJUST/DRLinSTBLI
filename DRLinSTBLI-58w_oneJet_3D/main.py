#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 09:48:16 2024

@author: yizhou
"""

import numpy as np
from stable_baselines3 import TD3,PPO
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_checker import check_env
from Env import openFoamEnv
import argparse
from torch import nn
from stable_baselines3.common.callbacks import  EvalCallback


if __name__ == '__main__':
    
    
    
    rootPath = '/Data/totalPBcLSTM-V3' # run path
    referenceP = (0.5*0.005275*1380.22*1380.22)*(-1) #reference dynamics pressure
    timeLimit = 5e-4# limiting the time of on episod
    deltaTime = 1e-5 # time of one episod
    scaleFactor =2000# scaling the action 
    n_obs = 168  # numbers of probes for the observation
    actLowLimit = -1 # low limit of the action
    actHighLimit = 1  # high limit of the action
    obsLowLimit = 0 # low limit of the observation
    obsHighLimit = np.inf  # high limit of the observation 
    alpha = 0.1 # smooth the action of the continuous time step (output = lastOutput + alpha*(netPrediction - lastOutput))
    batch_size=256 # size of selected batch for one training step 
    learning_starts=40 #  the step of randomly sampling before learning
    policy_kwargs = dict(net_arch=[256,256], activation_fn=nn.ReLU) #net architecture
    
    processor = 104 # the numbers of processors for the CFD computing
    gamma = 0.95 # discount factor
    learning_rate = 0.0001 #learning rate
    noise_std = 0.5 # standard deviation of noise
    seq_len = 50 # sequence length of one episode
    
    
    # Environment
    env = openFoamEnv(rootPath, actLowLimit, actHighLimit, 
                      obsLowLimit, obsHighLimit, timeLimit, 
                      deltaTime,referenceP,alpha,scaleFactor,
                      processor, n_obs, seq_len=seq_len)
    # number of action
    n_actions = env.action_space.shape[-1]

    # action noise
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=noise_std*np.ones(n_actions))
    model = TD3('MlpPolicy',env,action_noise=action_noise,verbose=1, seed = 0, gradient_steps=1,train_freq=1,
                learning_rate=learning_rate,policy_kwargs=policy_kwargs,batch_size=batch_size)
    
    # callBack
    eval_callback = EvalCallback(openFoamEnv(rootPath, actLowLimit, actHighLimit, 
                       obsLowLimit, obsHighLimit, timeLimit, 
                       deltaTime,referenceP,alpha,scaleFactor,
                       processor, spanwiseNodes,n_obs, seq_len=seq_len)
                                  , best_model_save_path='/'.join([rootPath,'logs/']),
                                  log_path='/'.join([rootPath,'logs/']),eval_freq=2000,
                                  deterministic=True,render=False,  n_eval_episodes=1)
    
    # learning and saveing
    model.learn(total_timesteps=7500)
    model.save('/'.join([rootPath,'model']))
    model.save_replay_buffer('/'.join([rootPath,'replayBuffer']))
    


    

    
