# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 19:56:25 2021

@author: orian
"""
import numpy as np
import value_free_environment as env
import value_free_agent as agent
import analysis_and_plot as anal

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
    U = [0,1,-1]
    trials_test = 500
    
    worlds = [[] for i in range(len(trials_list))]
    for n,trials in enumerate(trials_list):
        if omission or devaluation:
            trials = trials+trials_test
        environment = env.environment_free_operant(trials)
        worlds_same_train_duration = []
        for i in range(repetitions):
            print('simulation 4(VR):  ' + str(i+1)+'/'+str(repetitions))
            runs = agent.sim_Free_Operant_second(trials=trials,environment=environment)
            #runs = agent.sim_Free_Operant_minute(trials=trials,environment=environment)
            for t in range(trials):
                runs.run_agent(U, t, VI= VI, VR=VR)
            worlds_same_train_duration.append(runs)
        worlds[n] = worlds_same_train_duration
        return worlds

#worlds_VR = simulation_4_VR()
#anal.plot_4(worlds_VR, repetitions = 2) 


def simulation_4_VI(omission=False, devaluation=False):
    #in Variable interval(VI) schedule: reinforcers are 'baited' at vaiable intervals, and the fisrt press following baiting will lead to a reinforcer 
    U = [0.,1.0,-1.0]
    VI = 6
    repetitions = 10
    trials =10000   
    environment = env.environment_free_operant(trials)
    
    worlds = []
    for i in range(repetitions):
        print('simulation 4(VI):  ' + str(i+1)+'/'+str(repetitions))
        runs = agent.sim_Free_Operant_second(trials=trials,environment=environment)
        for t in range(trials):
            runs.run_agent(U,t,VI=VI)
        worlds.append(runs)        
    return worlds    


#simulation 4

#worlds_VR = simulation_4_VR(trials_list = [10000], repetitions = 10)
#anal.plot_4(worlds_VR, repetitions = 10)
#worlds_VI = simulation_4_VI()
#anal.plot_4(worlds_VI, repetitions = 2) 

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



            



























