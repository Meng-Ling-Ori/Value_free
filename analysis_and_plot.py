# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 22:03:10 2021

@author: orian
"""
import numpy as np
import matplotlib.pylab as plt
from scipy.signal import savgol_filter

def plot_1(worlds, repetitions):
    na = worlds[0].na
    trials = worlds[0].trials

        
    fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(nrows=5, figsize= (6,8))
    plt.suptitle('$w_g$:'+str(worlds[0].w_g)+' '+'$w_h$:'+str(worlds[0].w_h)+' '+ \
                     '$w_0$:'+str(worlds[0].w_0)+' '+r'$\alpha_H$'+str(worlds[0].alpha_H)+' '+ \
                         r'$\alpha_R$:'+str(worlds[0].alpha_R))
        
    # for probability of reinforcement
    ax1.plot(worlds[0].env.AM[:,0,1], color='blue', label='Action A')
    ax1.plot(worlds[0].env.AM[:,1,1], color='orange', label='Action B')
    ax1.set_title("probability of reinforcement")
    ax1.text(500,0.7, 'Action A', color = 'blue', fontweight = 'demibold')
    ax1.text(1300,0.7, 'Action B', color = 'orange', fontweight = 'demibold')
    
    # for Goal-Dierected Values(Q)
    Q = np.zeros((trials,na))
    for i in range(repetitions):
        Q += worlds[i].Q
    Q /= repetitions
    
    ax2.plot(Q[:,0], color='blue', label='Action A')
    ax2.plot(Q[:,1], color='orange', label='Action B')
    ax2.set_title("Goal-Dierected Values(Q)")
    
    # for Habit Strengths(H)
    H = np.zeros((trials,na))
    for i in range(repetitions):
        H += worlds[i].H
    H /= repetitions

    ax3.plot(H[:,0], color='blue', label='Action A')
    ax3.plot(H[:,1], color='orange', label='Action B')
    ax3.set_title("Habit Strengths(H)")

    # for Weight of Goal-Directed Control(W)
    W = np.zeros(trials)
    for i in range(repetitions):
        W += worlds[i].weights
    W /= repetitions
    W = 1 - W
    
    ax4.plot(W, color='blue', label='Action A')
    ax4.set_title("Weight of Goal-Directed Control(W)")

    # for Choice Probability
    P = np.zeros((trials,na))
    for i in range(repetitions):
        P += worlds[i].P
    P /= repetitions  

    ax5.plot(P[:,0], color='blue', label='Action A')
    ax5.plot(P[:,1], color='orange', label='Action B')
    ax5.set_title("Choice Probability")
    ax5.set_xlabel('trial number')

    ax = [ax1,ax2,ax3,ax4,ax5]
    for i in range(len(ax)):
        ax[i].set_ylim([0,1])
        ax[i].set_yticks([0,0.5,1])
        ax[i].set_xlim([0,trials])

    plt.tight_layout()
    plt.show()

def plot_2_0(worlds,repetitions,trials_phase1_list,trials_phase2):
    
    P = [[0] for i in range(trials_phase1_list.shape[0])]
    for i in range(trials_phase1_list.shape[0]):
        for j in range(repetitions):
            P[i] += worlds[i][j].P[:,0]
    
    plt.figure()
    for i in range(trials_phase1_list.shape[0]):
        P[i] /= repetitions    
        plt.plot(P[i])
    plt.title('probability of press over trials')
    plt.show()
 
def plot_2(worlds, repetitions, trials_phase1_list, trials_phase2):
        
    number_of_npress = np.zeros((trials_phase1_list.shape[0],2))
    for i, trials_phase1 in enumerate(trials_phase1_list):
        trials = trials_phase1 + trials_phase2
        for j in range(repetitions):
            number_of_npress[i,0] += worlds[i][j].actions[trials_phase1-1]
            number_of_npress[i,1] += worlds[i][j].actions[trials-1]
    number_of_press = repetitions - number_of_npress
    rate_of_press = number_of_press / repetitions
             
    plt.figure()
    plt.suptitle('$w_g$:'+str(worlds[0][0].w_g)+' '+'$w_h$:'+str(worlds[0][0].w_h)+' '+ \
                     '$w_0$:'+str(worlds[0][0].w_0)+' '+r'$\alpha_H$'+str(worlds[0][0].alpha_H)+' '+ \
                         r'$\alpha_R$:'+str(worlds[0][0].alpha_R))
           
    plt.plot(rate_of_press[:,0], color='blue', label='Press Rate: Training')
    plt.plot(rate_of_press[:,1], color='orange', label='Press Rate: Omission')
    plt.title('Overtraining Abolishes Sensitivity to Omission')
    plt.ylabel("Probability of Pressing(within 10 agents)")
    plt.xlabel('Number of Trials in Training Phase')
    plt.xlim([0,len(trials_phase1_list)])
    plt.xticks([0,5,10,15,20], [0,500,1000,1500,2000])
    plt.ylim([0,1])
    plt.yticks([0,0.2,0.4,0.6,0.8,1])
    plt.legend()
    plt.show()
    
    rate_of_action1 = np.zeros((trials_phase1_list.shape[0],2))
    for i, trials_phase1 in enumerate(trials_phase1_list):
        trials = trials_phase1 + trials_phase2
        for j in range(repetitions):
            rate_of_action1[i,0] += worlds[i][j].P[trials_phase1-1,0]
            rate_of_action1[i,1] += worlds[i][j].P[trials-1,0]
    rate_of_action1 /= repetitions
    
    plt.figure()
    plt.suptitle('$w_g$:'+str(worlds[0][0].w_g)+' '+'$w_h$:'+str(worlds[0][0].w_h)+' '+ \
                     '$w_0$:'+str(worlds[0][0].w_0)+' '+r'$\alpha_H$'+str(worlds[0][0].alpha_H)+' '+ \
                         r'$\alpha_R$:'+str(worlds[0][0].alpha_R))
           
    plt.plot(rate_of_action1[:,0], color='blue', label='Press Rate: Training')
    plt.plot(rate_of_action1[:,1], color='orange', label='Press Rate: Omission')
    plt.title('Overtraining Abolishes Sensitivity to Omission')
    plt.ylabel("Probability of Pressing(pi(a))")
    plt.xlabel('Number of Trials in Training Phase')
    plt.xlim([0,len(trials_phase1_list)])
    plt.xticks([0,5,10,15,20], [0,500,1000,1500,2000])
    plt.ylim([0,1])
    plt.yticks([0,0.2,0.4,0.6,0.8,1])
    plt.legend()
    plt.show()
       
def plot_3_0(worlds,repetitions,trials_phase1_list,trials_phase2,trials_phase3):
    
    P = [[0] for i in range(trials_phase1_list.shape[0])]
    for i in range(trials_phase1_list.shape[0]):
        for j in range(repetitions):
            P[i] += worlds[i][j].P[:,0]
    
    plt.figure()
    for i in range(trials_phase1_list.shape[0]):
        P[i] /= repetitions    
        plt.plot(P[i])
    plt.title('probability of press over trials')
    plt.show()
      
def plot_3(worlds, repetitions,trials_phase1_list,trials_phase2,trials_phase3):

    rate_of_press = np.zeros((trials_phase1_list.shape[0],3))
    
    for i, trials_phase1 in enumerate(trials_phase1_list):
        trials = trials_phase1 + trials_phase2 + trials_phase3
        rate_of_press[i,0] = 1 - np.mean([worlds[i][j].actions[trials_phase1-1] for j in range(repetitions)], axis=0)
        rate_of_press[i,1] = 1 - np.mean([worlds[i][j].actions[trials_phase1 + trials_phase2 - 1] for j in range(repetitions)], axis=0)
        rate_of_press[i,2] = 1 - np.mean([worlds[i][j].actions[trials-1] for j in range(repetitions)], axis=0)
#        for j in range(repetitions):
#            number_of_npress[i,0] += worlds[i][j].actions[trials_phase1-1]
#            number_of_npress[i,1] += worlds[i][j].actions[trials_phase1 + trials_phase2 - 1]
#            number_of_npress[i,2] += worlds[i][j].actions[trials-1]
#    number_of_press = repetitions - number_of_npress
#    rate_of_press = number_of_press / repetitions
             
    plt.figure()
    plt.suptitle('$w_g$:'+str(worlds[0][0].w_g)+' '+'$w_h$:'+str(worlds[0][0].w_h)+' '+ \
                     '$w_0$:'+str(worlds[0][0].w_0)+' '+r'$\alpha_H$'+str(worlds[0][0].alpha_H)+' '+ \
                         r'$\alpha_R$:'+str(worlds[0][0].alpha_R))
           
    plt.plot(rate_of_press[:,0], color='blue', label='Press Rate: Training')
    plt.plot(rate_of_press[:,1], color='orange', label='Press Rate: Devaluation')
    plt.plot(rate_of_press[:,2], color='grey', label='Press Rate: Distinction')
    plt.title('Overtraining Abolishes Sensitivity to Devaluation')
    plt.ylabel("Probability of Pressing(within 10 agents)")
    plt.xlabel('Number of Trials in Training Phase')
    plt.xlim([0,len(trials_phase1_list)])
    plt.xticks([0,5,10,15,20], [0,500,1000,1500,2000])
    plt.ylim([0,1])
    plt.yticks([0,0.2,0.4,0.6,0.8,1])
    plt.legend()
    plt.show()
    
    rate_of_action1 = np.zeros((trials_phase1_list.shape[0],3))
    for i, trials_phase1 in enumerate(trials_phase1_list):
        trials = trials_phase1 + trials_phase2 + trials_phase3
        for j in range(repetitions):
            rate_of_action1[i,0] += worlds[i][j].P[trials_phase1-1,0]
            rate_of_action1[i,1] += worlds[i][j].P[trials_phase1 + trials_phase2 - 1,0]
            rate_of_action1[i,2] += worlds[i][j].P[trials-1,0]
    rate_of_action1 /= repetitions
    
    plt.figure()
    plt.suptitle('$w_g$:'+str(worlds[0][0].w_g)+' '+'$w_h$:'+str(worlds[0][0].w_h)+' '+ \
                     '$w_0$:'+str(worlds[0][0].w_0)+' '+r'$\alpha_H$'+str(worlds[0][0].alpha_H)+' '+ \
                         r'$\alpha_R$:'+str(worlds[0][0].alpha_R))
           
    plt.plot(rate_of_action1[:,0], color='blue', label='Press Rate: Training')
    plt.plot(rate_of_action1[:,1], color='orange', label='Press Rate: Devaluation')
    plt.plot(rate_of_action1[:,2], color='grey', label='Press Rate: Distinction')
    plt.title('Overtraining Abolishes Sensitivity to Devaluation')
    plt.ylabel("Probability of Pressing(pi(a))")
    plt.xlabel('Number of Trials in Training Phase')
    plt.xlim([0,len(trials_phase1_list)])
    plt.xticks([0,5,10,15,20], [0,500,1000,1500,2000])
    plt.ylim([0,1])
    plt.yticks([0,0.2,0.4,0.6,0.8,1])
    plt.legend()
    plt.show() 
    

    
def plot_4(worlds, repetitions):


    plt.figure(figsize= (6,8))
    plt.suptitle('$w_g$:'+str(worlds[0][0].w_g)+' '+'$w_h$:'+str(worlds[0][0].w_h)+' '+ \
                     '$w_0$:'+str(worlds[0][0].w_0)+' '+r'$\alpha_H$'+str(worlds[0][0].alpha_H)+' '+ \
                         r'$\alpha_R$:'+str(worlds[0][0].alpha_R))
    
    # Outcomes(pellet rate, effort rate, habit strength responses over actions)
    

    try:
        Pellet_rate_true = (worlds[0][0].free_parameters['VR']/100)*worlds[0][0].actions_list
    except TypeError:
        Pellet_rate_true = 0.1 * worlds[0][0].actions_list # I haven't figured out how to calculate this for VI schedule

    Pellet_rate_agent = np.mean([worlds[0][i].R[5000,:,1] for i in range(repetitions)],axis=0) 
    Effort_rate_true =  worlds[0][0].rate_effort_true * (-1)
    Effort_rate_agent =  np.mean([worlds[0][i].R[5000,:,2] for i in range(repetitions)],axis=0) * (-1) 
    Habit_strength = np.mean([worlds[0][i].H[5000] for i in range(repetitions)],axis=0) 
#    data_responses = [worlds[i].actions[4000:5000] for i in range(repetitions)]
    
      #plt.hist(data, bins=40, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)    
    ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=2)
    ax1.plot(Pellet_rate_true, color='green',linestyle='-', label='Pellet rate: True')
    ax1.plot(Pellet_rate_agent, color='green',linestyle='--', label='Pellet rate: Agent')
    ax1.plot(Effort_rate_true, color='red',linestyle='-', label='Effort rate: True')
    ax1.plot(Effort_rate_agent, color='red',linestyle='--' ,label='Effort rate: Agent')
    ax1.plot(Habit_strength, color='orange',linestyle='--', label='Habit Strength')
    ax1.set_ylim([-20,30])
    ax1.set_yticks([-10,0,10,20,30])
    ax1.set_xlim([0,worlds[0][0].n_action_rate])
    ax1.set_xticks([0,worlds[0][0].n_action_rate/3,worlds[0][0].n_action_rate/1.5,worlds[0][0].n_action_rate])
    ax1.legend()
    if worlds[0][0].free_parameters['VR'] != None:
        ax1.set_title('Variable Ratio')
    else:
        ax1.set_title('Variable Interval')
    # smoothed press rate
    press_rate = np.mean([worlds[0][i].action_rate for i in range(repetitions)], axis=0)
    press_rate_smooth = savgol_filter(press_rate, 99, 2, mode= 'nearest')
    #press_rate_smooth = np.convolve(press_rate,'same')
    #press_rate_smooth = np.convolve(np.arange(0,10000,1),press_rate,'same')
    ax2 = plt.subplot2grid((6, 1), (2, 0))
    ax2.plot(press_rate_smooth, color='blue')
    #ax2.plot(press_rate, color='blue')
    ax2.set_ylim([0,100])
    ax2.set_yticks([0,50,100])
    ax2.set_xlim([0,10000])
    ax2.set_xticks([0,2000,4000,6000,8000,10000])
    ax2.set_title('smoothed press rate')
    
    #action-outcome contingency(g)
    gs = np.mean([worlds[0][i].gs for i in range(repetitions)], axis=0)
    ax3 = plt.subplot2grid((6, 1), (3, 0))
    ax3.plot(gs, color='green')
    ax3.set_ylim([0,1])
    ax3.set_yticks([0,0.5,1])
    ax3.set_xlim([0,10000])
    ax3.set_xticks([0,2000,4000,6000,8000,10000])
    ax3.set_title('action-outcome contingency(g)')
    
    #habitization(h)    
    hs = np.mean([worlds[0][i].hs for i in range(repetitions)], axis=0)
    ax4 = plt.subplot2grid((6, 1), (4, 0))
    ax4.plot(hs, color='orange')
    ax4.set_ylim([0,0.2])
    ax4.set_yticks([0,0.1,0.2])
    ax4.set_xlim([0,10000])
    ax4.set_xticks([0,2000,4000,6000,8000,10000]) 
    ax4.set_title('habitization(h)')
    
    #weight of goal-directed control(w)
    w = np.mean([worlds[0][i].weights for i in range(repetitions)], axis=0)
    w = 1-w
    ax5 = plt.subplot2grid((6, 1), (5, 0))
    ax5.plot(w, color='blue')
    ax5.set_ylim([0,1])
    ax5.set_yticks([0,0.5,1])
    ax5.set_xlim([0,10000])
    ax5.set_xticks([0,2000,4000,6000,8000,10000])     
    ax5.set_title('weight of goal-directed control(w)')
    
    plt.tight_layout()
    plt.show()    
    
    
    
    
    
    
    
    
    
    
    
    