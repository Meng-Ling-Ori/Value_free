# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 19:56:25 2021

@author: orian
"""
import numpy as np
import value_free_environment as env
import value_free_agent as agent
import analysis_and_plot as anal
from multiprocessing.pool import Pool
from functools import reduce

np.set_printoptions(threshold = 100000, precision = 5)

#1. 模拟2和模拟3的press的概率随着training duration的增加不会逐渐增加到1 最多到0.8
#2. 模拟4的计算 h始终值很小，g始终值很大，press rate（动作(a:0-150)选择概率）增加得非常快。

# Here are functions to run simulations and plot results.
# the class from 'Value_Free_Model': to implement all steps of value upload and action selection
# the functions from 'analysis_and_plot': to analyse results and plot

"""
simulation functions
(simulation 1 - 4)
"""

def simulation_1_reversal():
    trials_phase1 = 1000
    trials_phase2 = 5000
    trials = trials_phase1 + trials_phase2
    U = [0, 1.0, 0.1]
    repetitions = 10
    na = 2
    nm = 3
    
    AM = np.zeros((trials, na, nm)) #Probability of leading to different reinforcers from different actions
    AM[:,:,0] = 1
    AM[:trials_phase1,0,:] = [0.5, 0.5, 0]
    AM[trials_phase1:,1,:] = [0.5, 0.5, 0]
    
    environment = env.environment_two_alternative(trials,AM)   
    worlds = []    
    for i in range(repetitions):
        print('simulation 1:  ' + str(i+1)+'/'+str(repetitions))
        
        runs = agent.sim_two_alternative(trials=trials,environment=environment)
        
        for t in range(trials):
            runs.run_agent(U, t, reversal = True)        
        
        worlds.append(runs)        
    return worlds

#worlds_reversal = simulation_1_reversal()
#repetitions = 10
#anal.plot_1(worlds_reversal,repetitions) #plot


def simulation_2_omission():
#for knm in range(1):    
    # for omission
    trials_phase1_list = np.arange(100,2001,100)
    trials_phase2 = 500
    U = [0, 1.0, 0.1]
    repetitions = 10
    
    na=2
    nm=3
    
    worlds = [[] for i in range(trials_phase1_list.shape[0])] 
        # worlds[0]:training trials-100, worlds[1]:training trials-200..... 
        # worlds[0][0]:training trials-100 and the 1st repetition, worlds[0][1]:training trials-100 and the 2nd repetition
    
    for n,trials_phase1 in enumerate(trials_phase1_list): 
        print('simulation 2: training duration:  '+str(trials_phase1)+'-----'+str(n+1)+'/'+str(len(trials_phase1_list))+'runs')   
        
        trials = trials_phase1 + trials_phase2
        
        AM = np.zeros((trials,na,nm)) #Probability of leading to different reinforcers from different actions
        AM[:,:,0] = 1 
        AM[:trials_phase1,0,:] = [0.5, 0.5, 0] # training trials:press:0.75 no rewards/ 0.25 pellet
        AM[:trials_phase1,1,:] = [0, 0, 1] # training trials: hold-press: 100% leisure
        AM[trials_phase1:,1,:] = [0.5, 0.5, 0] #omission trials: hold-press:0.75 no rewards/ 0.25 pellet (during omission, hold-press results in 100% leisure, which is implemented in the run_agent)
        
        environment = env.environment_two_alternative(trials,AM)
        
        worlds_same_train_duration = []
        
        for i in range(repetitions):
            runs = agent.sim_two_alternative(trials=trials,environment=environment)
            
            for t in range(trials):
                runs.run_agent(U, t, omission = True, trials_phase1=trials_phase1)
            worlds_same_train_duration.append(runs)
        worlds[n] = worlds_same_train_duration       
    return worlds

#worlds_omission = simulation_2_omission()
#trials_phase1_list = np.arange(100,2001,100)
#trials_phase2 = 500

#anal.plot_2_0(worlds_omission,repetitions,trials_phase1_list,trials_phase2)
#anal.plot_2(worlds_omission,repetitions,trials_phase1_list,trials_phase2) #plot


def simulation_3_devaluation():
    
    trials_phase1_list = np.arange(100,2001,100)
    trials_phase2 = 500 #for devaluation phase, but they did not mention how many trials
    trials_phase3 = 500 #for extinction phase, but they did not mention how many trials
    na = 2
    nm = 3
    U1 = [0, 1.0, 0.1]
    U2 = [0,   0, 0.1] # for devaluation phase 
    U = [U1, U2]
    repetitions = 10 

    worlds = [[] for i in range(trials_phase1_list.shape[0])]
    for n,trials_phase1 in enumerate(trials_phase1_list): 
        print('simulation 3: training duration:  '+str(trials_phase1)+'-----'+str(n+1)+'/'+str(len(trials_phase1_list))+'runs')   
        
        trials = trials_phase1 + trials_phase2 + trials_phase3
        
        AM = np.zeros((trials,na,nm)) #Probability of leading to different reinforcers from different actions
        AM[:,:,0] = 1
        AM[:trials_phase1+trials_phase2,0,:] = [0.5,0.5,0]
        AM[:,1,:] = [0, 0, 1]  
        
        environment = env.environment_two_alternative(trials,AM)
        worlds_same_train_duration = []
        for i in range(repetitions):
            runs = agent.sim_two_alternative(trials=trials,environment=environment)
            for t in range(trials):
                runs.run_agent(U, t, devaluation = True, trials_phase1=trials_phase1, trials_phase2=trials_phase2)
            worlds_same_train_duration.append(runs)
        worlds[n] = worlds_same_train_duration       
    return worlds

#worlds_devaluation = simulation_3_devaluation()
#trials_phase1_list = np.arange(100,2001,100)
#trials_phase2 = 500 
#trials_phase3 = 500 

#anal.plot_3_0(worlds_devaluation, repetitions,trials_phase1_list,trials_phase2,trials_phase3)
#anal.plot_3(worlds_devaluation, repetitions,trials_phase1_list,trials_phase2,trials_phase3)


def simulation_4(trials_list,repetitions,omission=False, devaluation=False, VI = None, VR = None):
    #in Variable ration(VR) schedule: the probability of receiving a reinforcer is constant after each lever press
    U = [0.1,1,-1]
    trials_test = 500
    
    worlds = [[] for i in range(len(trials_list))]
    for n,trials in enumerate(trials_list):
        if omission or devaluation:
            trials = trials+trials_test
        environment = env.environment_free_operant(trials)
        worlds_same_train_duration = []
        for i in range(repetitions):
            if VR != None:
                print('simulation 4(VR):  ' + str(i+1)+'/'+str(repetitions))
            else:
                print('simulation 4(VI):  ' + str(i+1)+'/'+str(repetitions))
            runs = agent.sim_Free_Operant_second(trials=trials,environment=environment)
            #runs = agent.sim_Free_Operant_minute(trials=trials,environment=environment)
            for t in range(trials):
                runs.run_agent(U, t, VI= VI, VR=VR)
            worlds_same_train_duration.append(runs)
        worlds[n] = worlds_same_train_duration
        return worlds

#worlds_VR = simulation_4_VR()
#anal.plot_4(worlds_VR, repetitions = 2) 


def simulation_5_MF_MB(trials, repetitions):
    # two armed bandit task
    environment = env.environment_two_armedbandit(trials) 
    alpha_MF_list = np.arange(0.2,1,0.1) # 0.2-1 #model free
    alpha_MB_list = np.arange(0.2,1,0.1) #0.2-1 #model based
    theta_MF_list = np.arange(0,10,0.1) #0-10
    theta_MB_list = np.arange(0,10,0.1) #0-10
    w_list = np.arange(0,1,0.1) #0-1
    
    test_number = 200
    worlds = [[] for i in range(test_number)]
    par_list = np.zeros((test_number,5))
    for k in range(test_number):
        par_list[k] = [np.random.choice(alpha_MF_list), np.random.choice(alpha_MB_list), \
            np.random.choice(theta_MB_list),np.random.choice(theta_MF_list),np.random.choice(w_list)]
        
        print('simulation 5 (MF/MB):'+str(k+1)+'/'+str(int(len(worlds))))
        worlds_same_par = []                                       
        for i in range(repetitions):
            runs = agent.sim_two_armedbandit_MF(trials,environment,*par_list[k])
        
            for t in range(trials):
                runs.run_agent(t)                
            worlds_same_par.append(runs)
        worlds[k] = worlds_same_par
    return worlds

def simulation_5_perseverativ(trials, repetitions):
    # two armed bandit task
    environment = env.environment_two_armedbandit(trials)
    alpha_H_list = np.arange(0.5,0.7,0.1) #0.5-0.7
    alpha_R_list = np.arange(0.5,0.7,0.1) #0.5-0.7
    theta_h_list = np.arange(1,3,0.1) #1-3
    theta_g_list = np.arange(3,6,0.1) #3-6
    w_h_list = np.arange(1,3,0.1) #1-3
    w_g_list = np.arange(8,12,0.1) #8-12
    w_0_list = np.arange(1,3,0.1) #1-3
    
    U = [0,1]
    test_number = 50
    worlds = [[] for i in range(test_number)]
    par_list = np.zeros((test_number,7))
    for k in range(test_number):
        par_list[k] = [np.random.choice(alpha_H_list), np.random.choice(alpha_R_list), \
            np.random.choice(theta_h_list),np.random.choice(theta_g_list), \
                np.random.choice(w_h_list),np.random.choice(w_g_list),np.random.choice(w_0_list)]
        
        print('simulation 5 (perseverativ):'+str(k+1)+'/'+str(int(len(worlds))))
        worlds_same_par = []                                       
        for i in range(repetitions):
            runs = agent.sim_two_armedbandit_per(trials,environment,*par_list[k])
                                                         
            for t in range(trials):
                runs.run_agent(t,U)                
            worlds_same_par.append(runs)
        worlds[k] = worlds_same_par
    return worlds    
    

#%%
def simulation_5_MF_MB_fake(trials, repetitions):
    # two armed bandit task
    environment = env.environment_two_armedbandit(trials) 
    alpha_MF_list = [0.2,0.6,1] # 0.2-1 #model free
    alpha_MB_list = [0.2,0.6,1] #0.2-1 #model based
    theta_MF_list = [1,5,9] #0-10
    theta_MB_list = [1,5,9] #0-10
    w_list = [0.1,0.5,0.9] #0-1
    a,b,c,d,e = len(alpha_MB_list), len(alpha_MF_list), len(theta_MB_list), len(theta_MF_list), len(w_list)
    worlds = [[] for i in range(a*b*c*d*e)]
    
    for aMF,alpha_MF in enumerate(alpha_MF_list):
        for aMB,alpha_MB in enumerate(alpha_MB_list):
            for thMF,theta_MF in enumerate(theta_MF_list):
                for thMB,theta_MB in enumerate(theta_MB_list):
                    for k,w in enumerate(w_list):
                        worlds_same_par = [] 
                        print('simulation 5 (MF/MB):'+str(b*c*d*e*aMF+c*d*e*aMB+d*e*thMF+e*thMB+k+1)+'/'+str(int(len(worlds))))
                                               
                        for i in range(repetitions):
                            runs = agent.sim_two_armedbandit_MF(trials,environment,\
                                         alpha_MF,alpha_MB,theta_MB,theta_MF,w)
        
                            for t in range(trials):
                                runs.run_agent(t)                
                            worlds_same_par.append(runs)
                        worlds[(b*c*d*e*aMF+c*d*e*aMB+d*e*thMF+e*thMB+k)] = worlds_same_par
    return worlds

def simulation_5_perseverativ_fake(trials, repetitions):
    # two armed bandit task
    environment = env.environment_two_armedbandit(trials)
    alpha_H_list = [0.5,0.6,0.7] #0.5-0.7
    alpha_R_list = [0.5,0.6,0.7] #0.5-0.7
    theta_h_list = [1,2,3] #1-3
    theta_g_list = [3,4.5,6] #3-6
    w_h_list = [1,2,3] #1-3
    w_g_list = [8,10,12] #8-12
    w_0_list = [1,2,3] #1-3
    
    U = [0,1]
    a,b,c,d,e,f,g = len(alpha_H_list), len(alpha_R_list), len(theta_h_list), len(theta_g_list), len(w_h_list), len(w_g_list), len(w_0_list)
    worlds = [[] for i in range(a*b*c*d*e*f*g)]
    for a1,alpha_H in enumerate(alpha_H_list):
        for a2,alpha_R in enumerate(alpha_R_list):
            for th,theta_h in enumerate(theta_h_list):
                for tg,theta_g in enumerate(theta_g_list):
                    for wh,w_h in enumerate(w_h_list):
                        for wg,w_g in enumerate(w_g_list):
                            for w0,w_0 in enumerate(w_0_list):
                                worlds_same_par = [] 
                                print('simulation 5 (perseverativ):'+str(b*c*d*e*f*g*a1+c*d*e*f*g*a2+d*e*f*g*th+e*f*g*tg+f*g*wh+g*wg+w0+1)+'/'+str(int(len(worlds))))
                                               
                                for i in range(repetitions):
                                    runs = agent.sim_two_armedbandit_per(trials,environment,\
                                         alpha_H, alpha_R,theta_h,theta_g,w_h,w_g,w_0)
        
                                    for t in range(trials):
                                        runs.run_agent(t,U)                
                                    worlds_same_par.append(runs)
                                worlds[(b*c*d*e*f*g*a1+c*d*e*f*g*a2+d*e*f*g*th+e*f*g*tg+f*g*wh+g*wg+w0)] = worlds_same_par    
    return worlds

        
  
def simulation_4_min(trials_list,repetitions,omission=False, devaluation=False, VI = None, VR = None):
    #in Variable ration(VR) schedule: the probability of receiving a reinforcer is constant after each lever press
    U = [-1,1]
    trials_test = 500
    
    worlds = [[] for i in range(len(trials_list))]
    for n,trials in enumerate(trials_list):
        if omission or devaluation:
            trials = trials+trials_test
        environment = env.environment_free_operant(trials)
        worlds_same_train_duration = []
        for i in range(repetitions):
            if VR != None:
                print('simulation 4(VR):  ' + str(i+1)+'/'+str(repetitions))
            else:
                print('simulation 4(VI):  ' + str(i+1)+'/'+str(repetitions))
            runs = agent.sim_Free_Operant_minute(trials=trials,environment=environment)
            #runs = agent.sim_Free_Operant_minute(trials=trials,environment=environment)
            for t in range(trials):
                runs.run_agent(U, t, VI= VI, VR=VR)
            worlds_same_train_duration.append(runs)
        worlds[n] = worlds_same_train_duration
        return worlds
#%%
"""
run
"""
repetitions = 10

#simulation1
worlds_reversal = simulation_1_reversal()
anal.plot_1(worlds_reversal,repetitions) #plot


#simulation2
worlds_omission = simulation_2_omission()
trials_phase1_list = np.arange(100,2001,100)
trials_phase2 = 500


#anal.plot_2_0(worlds_omission,repetitions,trials_phase1_list,trials_phase2) #show the probability of press for all training duration
anal.plot_2(worlds_omission,repetitions,trials_phase1_list,trials_phase2) 
    # 1 for proportion of selected action of the last trial of 10 repetitions
    # 2 for average probability of actions of 10 repetitions(Pi(a))


#simulation3
worlds_devaluation = simulation_3_devaluation()
trials_phase1_list = np.arange(100,2001,100)
trials_phase2 = 500 
trials_phase3 = 500 
#anal.plot_3_0(worlds_devaluation, repetitions,trials_phase1_list,trials_phase2,trials_phase3)
anal.plot_3(worlds_devaluation, repetitions,trials_phase1_list,trials_phase2,trials_phase3) 
    # 1 for proportion of selected action of the last trial of 10 repetitions
    # 2 for average probability of actions of 10 repetitions(Pi(a))


#simulation 4
worlds_VR = simulation_4(trials_list = [10000], repetitions = 10, VR=10)
anal.plot_4(worlds_VR, repetitions = 10)

worlds_VI = simulation_4(trials_list = [10000], repetitions = 10, VI=6)
anal.plot_4(worlds_VI, repetitions = 10)


#worlds_VR_min = simulation_4_min(trials_list = [10000], repetitions = 10, VR=10)
#anal.plot_4(worlds_VR_min, repetitions = 10)
            
#%%  
#simulation 5        
worlds_MF = simulation_5_MF_MB(trials=10000,repetitions=1)

worlds_per = simulation_5_perseverativ(trials=10000,repetitions=1)
#%%
beta_list_MF = anal.plot_5(worlds_MF, repetitions=1,MF=True)

beta_list_per = anal.plot_5(worlds_per, repetitions=1)




#%%
# to save data
import pickle
import numpy as np

data = beta_list_per
# wb save data
data_output = open('data_beta_per.pkl','wb')
pickle.dump(data,data_output)
data_output.close()
#%%
# rb load data
data_input = open('data.pkl','rb')
read_data = pickle.load(data_input)
data_input.close()















