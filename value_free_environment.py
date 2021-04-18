# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 02:14:54 2021

@author: orian
"""
import numpy as np
import random


class environment_two_alternative(object):
    def __init__(self,trials,AM,nm=3):
        
        self.nm = nm
        self.AM = AM
        
        self.reinforcers = np.zeros((trials), dtype = int)
        self.rewards = np.zeros((trials))

        
    def obtained_reinforcement(self,t,action):
        reinforcer = np.random.choice(range(self.nm), p = self.AM[t,action,:])
        self.reinforcers[t] = reinforcer
        
        return reinforcer
    
    def obtained_rewrads(self,t,U,omission,trials_phase1,action):
        if not omission:
            reward = U[self.reinforcers[t]]

        else:
            if t > (trials_phase1-1) and action == 1:
                reward = U[self.reinforcers[t]] + U[2]
            else:
                reward= U[self.reinforcers[t]]            
        self.rewards[t] = reward
        
class environment_free_operant(object):
    def __init__(self,trials,nm=3):
        
        self.nm = nm
        
        self.reinforcers = np.zeros((trials,nm), dtype = int)
        self.effort_rate = np.zeros(trials)
        self.rewards = np.zeros((trials))
    
    def obtained_reinforcement(self,t,action,VR,VI,omission):
        reinforcer = np.zeros(self.nm) # the number of obtained reinforcer: leisure, pellets and effort
        if VR != None: # for VR schedule
            if not omission:
                if action[0] != 1: # if press, 10% obtained 'pellets'
                    reinforcer[1] = sum([np.random.choice(range(self.nm-1), p = [1-VR/100, VR/100]) for i in range(np.where(action ==1)[0][0])])
                else: # if not press, 100% get 'leisure', 
                    reinforcer[0] = 1
                reinforcer[2] = self.effort_rate[t]/ 60  # effort in one second           
                    
            else:
                if action[0] == 1: #if press, reinforcer = non-rewards
                    number_of_reinforcer = 1
                else: # if press, 10% obtained 'pellets'
                    number_of_reinforcer = sum([np.random.choice(range(self.nm-1), p = [1-VR/100, VR/100]) for i in range(np.where(action ==1)[0][0])])                

        
#        else: # for VI schedule
#            if action[0] == 1: # if not press, reinforcer = non-rewards
#                reinforcer = 0
#            elif max(self.reinforcers[1]) == 0 : # if press and haven't got pellets , get 'pellets'
#                reinforcer = 1
#            elif t - np.where(self.reinforcers[1] == 1)[0][-1] >= VI: # if press but already got 'pellets' in previous trial, and previous six trial did not get 'pellets', get 'pellets'
#                reinforcer = 1
#            else:
#                reinforcer = 0
#            number_of_reinforcer = 1
            
#        self.reinforcers[t,reinforcer] = number_of_reinforcer
        self.reinforcers[t] = reinforcer
        
        return reinforcer
    
    def obtained_effort(self,t,action_rate):            
        ### obtained effort                        
        effort_rate = 2*(10**(-3))*action_rate + 6*(10**(-4))*action_rate**2
        self.effort_rate[t] = effort_rate
        
        return effort_rate
    
    def obtained_rewrads(self,t,U):
        ### obtained rewards
        self.rewards[t] = U[1]*self.reinforcers[t,1] + U[2]*self.effort_rate[t]
