# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 19:21:54 2021

@author: orian
"""

"""
There are 2 classes here: each corresponding to a framework.
They are responsible for iterating the number of steps, updating the values and selecting the actions. 
Finally, the results of R, H, g, s, weights of the habit system, etc., obtained for each step are returned
"""
import numpy as np
import random


np.set_printoptions(threshold = 100000, precision = 5)

class sim_two_alternative(object):
    def __init__(self, trials, environment,
                 na=2, nm=3, alpha_H=0.001, alpha_R=0.01, 
                 w_g=5, w_h=5, w_0=1, theta_g=5, theta_h=5): 
        
        self.env = environment
        
        self.na = na
        self.nm = nm
        self.trials = trials

        self.H = np.zeros((trials,na)) #habitual strength
        self.R = np.zeros((trials,na,nm)) #expectation of reinforcers after action
        self.Q = np.zeros((trials,na)) #expectation value of action 
        self.D = np.zeros((trials,na)) #overall drive
        self.P = np.zeros((trials,na)) #pi(a)
        
        self.gs = np.zeros((trials)) #action-outcome contingency(here only consider m_1)
        self.hs = np.zeros((trials)) #overall habitization
        self.weights = np.zeros((trials))
        self.actions = np.zeros((trials), dtype = int)
        self.reinforcers = np.zeros((trials), dtype = int)
        
        self.alpha_H = alpha_H
        self.alpha_R = alpha_R
        self.w_g = w_g # scaling parameter controlling the relatice strength of the goal-directed system
        self.w_h = w_h # scaling parameter controlling the relatice strength of the habitual system
        self.w_0 = w_0 # bias parameter
        self.theta_g = theta_g #step-size parameter, which determines the rate of change(habitual strength)
        self.theta_h = theta_h #step-size parameter, determines the rate of change(expectation of reinforcers)
        
    
    def run_agent(self, U, current_trial, 
                  reversal = False, omission = False, devaluation =  False,
                  trials_phase1 = None, trials_phase2 = None, trials_phase3 = None):
        
        t = current_trial 
        
        if devaluation:
            if t <= (trials_phase1-1) or t >= trials_phase2 + trials_phase1:
                U = [0, 1.0, 0.1]
            else:
                U = [0,   0, 0.1]
        
        if t == 0:
            self.set_initial()
        else:
            self.update_habitual_system(t) # H
        
            self.update_goal_directed_system(t,omission,trials_phase1,U)  # R,Q
        
        self.update_arbiter(t) # g,h,w,D
        
        self.update_pi_a(t) #action selection probability
        
        self.actioin_selection(t) #select action according to the probability
    
        self.reinforcers[t] = self.env.obtained_reinforcement(t,self.actions[t]) #obtained reinforcer
        
        self.env.obtained_rewrads(t,U,omission,trials_phase1,self.actions[t])
    
    def set_initial(self):
        self.H[:,:] = 0
        self.R[:,:,:] = 0
        self.Q[:,:] = 0
                
    def update_habitual_system(self,t):
        ### habitual strength       
        a = self.actions[t-1]
            
        a_t = np.zeros(self.na) # selected action at the last step
        a_t[a] = 1
        self.H[t,:] = self.H[t-1,:] + self.alpha_H*(a_t - self.H[t-1,:])  
    
    def update_goal_directed_system(self,t,omission,trials_phase1,U):
        
        m = self.reinforcers[t-1]
        a = self.actions[t-1]
            
        ###expectation of reinforcers after action
        r_t = np.zeros((self.nm))
        r_t[m] = 1 # magnitude of reinforcers following the selected action           
            
        if omission: 
            if t-1 > (trials_phase1-1) and a == 1: # when phase 2 and choose action b 
                r_t[2] = 1
                
        self.R[t] = self.R[t-1]
        self.R[t,a] = self.R[t-1,a] + self.alpha_R*(r_t - self.R[t-1,a])
        ###expectation value of action 
        self.Q[t] = np.sum([U[i]*self.R[t,:,i] for i in range(self.nm)], axis = 0)        

    def update_arbiter(self,t):
        ### arbiter
        # action-outcome contingency(here only consider m_1)
        if t == 0:
            self.gs[t] = 0
        else:
            self.gs[t] = np.sqrt(sum(self.P[t-1,i]*(self.R[t,i,1] \
                            - sum(self.P[t-1,j]*self.R[t,j,1] for j in range(self.na)))**2 for i in range(self.na)))
        
        #overall habitization
        self.hs[t] = np.sqrt(sum((self.H[t,i] - np.mean(self.H[t,:]))**2 for i in range(self.na)))
    
        #weight of habitual system
        self.weights[t] = 1/(1 + np.exp(self.w_g * self.gs[t] - self.w_h * self.hs[t] + self.w_0))
    
        #overall drive
        w = self.weights[t]
        self.D[t] = w * self.theta_h * self.H[t] + (1-w) * self.theta_g * self.Q[t]        
        
    def update_pi_a(self,t):
        #pi(a), selects actions accrording to a softmax on D
        self.P[t] = np.array([np.exp(self.D[t,i])/sum(np.exp(self.D[t,:])) for i in range(self.na)])

    def actioin_selection(self,t):
        #action selection
        self.actions[t] = np.random.choice(range(self.na), p = self.P[t])


class sim_Free_Operant_second(object):
    def __init__(self, trials, environment, trials_test = 500, 
                 actions_list=np.arange(0,150,2),
                 nm=3, na=2, ni_R=4, ni_H = 30,
                 alpha_H=10**(-5), alpha_R=10**(-1), 
                 w_g=6, w_h=15, w_0=1, theta_g=20, theta_h=10**3): 
        
        self.env = environment         
        
        self.trials = trials
        self.trials_test = trials_test
        self.nm = nm
        self.na = na
        self.actions_list = actions_list
        self.n_action_rate = len(actions_list)
        
        self.H = np.zeros((trials,self.n_action_rate)) 
        self.R = np.zeros((trials,self.n_action_rate,nm)) 
        self.Q = np.zeros((trials,self.n_action_rate)) 
        self.D = np.zeros((trials,self.n_action_rate)) 
        self.P = np.zeros((trials,self.n_action_rate)) 
        
        self.m_a  = np.array([np.exp(-i/5) for i in self.actions_list]) # action density
        self.b_t = np.zeros((trials,nm,ni_R))
        self.c_t = np.zeros((trials,ni_H))
        self.gs = np.zeros((trials))
        self.hs = np.zeros((trials)) 
        self.weights = np.zeros((trials))
        self.actions = np.zeros((trials,21), dtype = int) # the number of press within one second
            # with the max rate is 150 and with uniform speed, the max number of press is 3  
        self.action_rate  = np.zeros(trials)
        self.rate_effort_true = 2*(10**(-3))*actions_list + 6*(10**(-4))*(actions_list)**2 # for effort 
        self.reinforcers = np.zeros((trials,nm))
        self.reinforcers_rate = np.zeros((trials,nm)) # the rates of getting pellet or non-rewards or effort of each trial
        self.effort_rate = np.zeros(trials) # the rates of effort of each trial(=action)


        self.alpha_H = alpha_H
        self.alpha_R = alpha_R        
        self.ni_R = ni_R
        self.ni_H = ni_H
        self.w_g = w_g
        self.w_h = w_h
        self.w_0 = w_0
        self.theta_h = theta_h
        self.theta_g = theta_g
        self.free_parameters = {} 
        
        self.phi = np.zeros((ni_R,self.n_action_rate))
        for i in range(ni_R):
            for j,k in enumerate(self.actions_list):
                self.phi[i,j] = ((k-75)/75)**i


        self.chi = np.zeros((ni_H,self.n_action_rate))
        for i in range(ni_H):
            for j,k in enumerate(self.actions_list):
                self.chi[i,j] = np.exp( -((k-5*i)**2)/(2*5**2))
        
        
    def run_agent(self, U, current_trial, VR = None, VI = None, omission = False, devaluation = False):
        

        self.free_parameters.update({'VR':VR,'VI':VI, 'omission':omission, 'devaluation':devaluation})
        
        t = current_trial 
        
        if devaluation:
            if t < self.trials - self.trials_test:
                U = [0.1,1,-1]
            else:
                U = [0.1,0,-1]       
 
        if t == 0:
            self.set_initial()
        else:
            self.update_habitual_system(t) # H
        
            self.update_goal_directed_system(t,U)  # R,Q
        
        self.update_arbiter(t) # g,h,w,D
        
        self.update_pi_a(t) #actions rate selection probability
        
        self.action_rate_selection(t) #select actions rate according to the probability
#        self.action_selection(t) #select action accroding to the actions rate
        self.action_selection_poisson(t)
        
        self.effort_rate[t] = self.env.obtained_effort(t,self.action_rate[t]) #/60
    
        reinforcer = self.env.obtained_reinforcement(t,self.actions[t],VR,VI,omission) #obtained reinforcer
        self.reinforcers[t] = reinforcer
               
        self.reinforcers_rate[t] = self.reinforcers[t]
        self.reinforcers_rate[t,1:3] *=60# the reinforcers rate of leisure keep 1
        
#        self.env.obtained_rewrads(t,U)
        
      
    def set_initial(self):
        self.H = np.zeros((self.trials,self.n_action_rate)) 
        self.R = np.zeros((self.trials,self.n_action_rate,self.nm)) 
        self.Q = np.zeros((self.trials,self.n_action_rate)) 

    def update_habitual_system(self,t):
        tau = np.where(self.actions_list == self.action_rate[t-1])[0][0] #the position of the selected actions rate
        self.c_t[t] = self.c_t[t-1] + self.alpha_H * ( 1 - self.H[t-1,tau]) * self.chi[:,tau] \
                                    + self.alpha_H * np.sum(( 0 - self.H[t-1]) * self.chi,axis=1) \
                                    - self.alpha_H * ( 0 - self.H[t-1,tau]) * self.chi[:,tau]
#        for i in range(self.ni_H):
#            for j in range(self.n_action_rate):
#                if j == tau:
#                    self.c_t[t,i] += self.alpha_H*(1-self.H[t-1,j])*self.chi[i,j]
#                else:
#                    self.c_t[t,i] += self.alpha_H*(0-self.H[t-1,j])*self.chi[i,j]
        self.H[t] = np.dot(self.c_t[t],self.chi).T
        #c_t:(trials,ni_H) ; chi(ni_H,n_action_rate)
            
    def update_goal_directed_system(self,t,U):
        tau = np.where(self.actions_list == self.action_rate[t-1])[0][0] #the position of the selected actions rate
                       
        r_t = self.reinforcers_rate[t-1]
        self.b_t[t] = self.b_t[t-1] + self.alpha_R * ((r_t - self.R[t-1,tau]).reshape(self.nm,1) * self.phi[:,tau].reshape(1,self.ni_R))

#        for i in range(self.ni_R):
#            for k in range(self.nm):
#                self.b_t[t,k,i] += self.alpha_R*(r_t[k] - self.R[t-1,tau,k])*self.phi[i,tau]

        self.R[t] = np.dot(self.b_t[t],self.phi).T 
        self.Q[t] = np.dot(self.R[t], np.array(U))
#        self.Q[t] = np.dot(R_60, np.array(U))
                
    def update_arbiter(self,t):

        if t == 0:
            self.gs[t] = 0
        else:
            self.gs[t] = np.sqrt(np.sum(self.P[t-1]*(self.R[t,:,1] - np.sum(self.P[t-1]*self.R[t,:,1]))**2))
#            self.gs[t] = np.sqrt(np.sum(self.P[t-1]*(R_60[:,1] - np.sum(self.P[t-1]*R_60[:,1]))**2))
        self.hs[t] = np.sqrt(np.sum((self.H[t] - np.mean(self.H[t]))**2))
            
        ### update weight
        self.weights[t] = 1/(1 + np.exp(self.w_g * self.gs[t] \
                                        - self.w_h * self.hs[t] \
                                        + self.w_0))

        w = self.weights[t]
        
        ### update overall drive
        self.D[t] = w * self.theta_h * self.H[t] + (1-w) * self.theta_g * self.Q[t]
            
    def update_pi_a(self,t):  
        
        ### update pi(a)
        self.P[t,:] = np.exp(self.m_a*self.D[t]) / np.sum(np.exp(self.m_a*self.D[t]))

    def action_rate_selection(self,t):     
        ### actions rate selection    
        self.action_rate[t] = self.actions_list[np.random.choice(range(self.n_action_rate), p = self.P[t])]
    
#    def action_selcetion_01
    def action_selection(self,t):
        ### action selection
        p_press = np.zeros(4)
        #p_press: the probability that agent press 0-3 time within a second
        if self.action_rate[t] >= 120:
            p_press[2] = 1 - (self.action_rate[t] - 120)/60 
            p_press[3] = (self.action_rate[t] - 120)/60 
        elif self.action_rate[t] >= 60:
            p_press[1] = 1 - (self.action_rate[t] - 60)/60
            p_press[2] = (self.action_rate[t] - 60)/60
        else:
            p_press[0] = 1-self.action_rate[t] / 60
            p_press[1] = self.action_rate[t] / 60
        
        
        action = np.random.choice(range(4), p = p_press)
        self.actions[t,action] = 1
        
    def action_selection_poisson(self,t):
        max_press = 20+1  # assume the range of press number within a second is 0-20
        p_press = np.zeros(max_press) # probability of pressing number
        avg_a = self.action_rate[t] / 60 # average press number within one second
        for i in range(max_press):
            p_press[i] = ((avg_a**i)/np.math.factorial(i) ) * np.exp(-avg_a)
            
        action = np.random.choice(range(max_press), p = p_press)
        self.actions[t,action] = 1


class sim_two_armedbandit_MF(object):
    def __init__(self, trials, environment,\
                 alpha_MF, alpha_MB,theta_MB,theta_MF,w,\
                 na=2, nm=2, ):
        self.env = environment
        self.na = na
        self.nm = nm
        self.trials = trials
        
        self.alpha_MF = alpha_MF
        self.alpha_MB = alpha_MB
        self.theta_MB = theta_MB
        self.theta_MF = theta_MF
        self.w = w
        
        self.H = np.zeros((trials,na))
        self.Q = np.zeros((trials,na))
        self.D = np.zeros((trials,na))
        self.P = np.zeros((trials,na))
        
        self.action = np.zeros((trials,na), dtype = int)
        self.reinforcer = np.zeros((trials,nm), dtype = int)
        self.free_parameters = {'alpha_MF': alpha_MF,'alpha_MB':alpha_MB,'theta_MB':theta_MB,'theta_MF':theta_MF,'w':w}
        
        
        
        
    def run_agent(self,current_trial):
        t = current_trial 
        if t == 0:
            self.set_initial()
        else:
            self.update_habitual_system(t) # H
        
            self.update_goal_directed_system(t)  # Q
        

        self.update_overall_driver(t) # D
        
        self.update_pi_a(t) #actions probability update
        
        action = self.action_selection(t) # choose action
        
        reinforcer = self.env.obtained_reinforcement(t,action)  
        self.reinforcer[t,reinforcer] = 1           
        
    def set_initial(self):
        self.H = np.zeros((self.trials,self.na)) 
        self.Q = np.zeros((self.trials,self.na))
    
    def update_habitual_system(self,t):
        a = np.where(self.action[t-1]==1)[0]
        r_t = np.where(self.reinforcer[t-1]==1)[0]
        self.H[t] = self.H[t-1]
        self.H[t,a] = self.H[t-1,a] + self.alpha_MF * (r_t - self.H[t-1,a])
    
    def update_goal_directed_system(self,t):
        a = np.where(self.action[t-1]==1)[0]
        r_t = np.where(self.reinforcer[t-1]==1)[0]
        self.Q[t] = self.Q[t-1]
        self.Q[t,a] = self.Q[t-1,a] + self.alpha_MB * (r_t - self.Q[t-1,a])

    def update_overall_driver(self,t):
        w = self.w
        theta_MF = self.theta_MF
        theta_MB = self.theta_MB
        self.D[t] = w * theta_MF * self.H[t] + (1-w) * theta_MB * self.Q[t]
        
    def update_pi_a(self,t):
        self.P[t] = np.array([np.exp(self.D[t,i])/sum(np.exp(self.D[t,:])) for i in range(self.na)])

    def action_selection(self,t):
        action = np.random.choice(range(self.na), p = self.P[t])
        self.action[t,action] = 1 
        return action
    
class sim_two_armedbandit_per(object):
    def __init__(self, trials, environment,\
                 alpha_H, alpha_R,theta_h,theta_g,w_h,w_g,w_0,\
                 na=2, nm=2, ):
        self.env = environment
        self.na = na
        self.nm = nm
        self.trials = trials
        
        self.H = np.zeros((trials,na)) #habitual strength
        self.R = np.zeros((trials,na,nm)) #expectation of reinforcers after action
        self.Q = np.zeros((trials,na)) #expectation value of action 
        self.D = np.zeros((trials,na)) #overall drive
        self.P = np.zeros((trials,na)) #pi(a)
        
        self.gs = np.zeros((trials)) #action-outcome contingency(here only consider m_1)
        self.hs = np.zeros((trials)) #overall habitization
        self.weights = np.zeros((trials))
        self.action = np.zeros((trials,na), dtype = int)
        self.reinforcer = np.zeros((trials,nm), dtype = int)
        
        self.alpha_H = alpha_H
        self.alpha_R = alpha_R
        self.w_g = w_g # scaling parameter controlling the relatice strength of the goal-directed system
        self.w_h = w_h # scaling parameter controlling the relatice strength of the habitual system
        self.w_0 = w_0 # bias parameter
        self.theta_g = theta_g #step-size parameter, which determines the rate of change(habitual strength)
        self.theta_h = theta_h #step-size parameter, determines the rate of change(expectation of reinforcers)

        self.free_parameters = {'alpha_H': alpha_H,'alpha_R':alpha_R,'theta_h':theta_h,'theta_g':theta_h,'w_h':w_h,'w_g':w_g,'w_0':w_0}
        
    def run_agent(self,current_trial,U):
        t = current_trial 
        if t == 0:
            self.set_initial()
        else:
            self.update_habitual_system(t) # H
        
            self.update_goal_directed_system(t,U)  # R,Q
        
        self.update_arbiter(t) # g,h,w,D
        
        self.update_pi_a(t) #action selection probability
        
        
        action = self.action_selection(t) # choose action
        
        reinforcer = self.env.obtained_reinforcement(t,action)  
        self.reinforcer[t,reinforcer] = 1           
        
    def set_initial(self):
        self.H = np.zeros((self.trials,self.na))
        self.R = np.zeros((self.trials,self.na,self.nm))
        self.Q = np.zeros((self.trials,self.na))
    
    def update_habitual_system(self,t):
        a_t = self.action[t-1]        
        self.H[t,:] = self.H[t-1,:] + self.alpha_H*(a_t - self.H[t-1,:])  
    
    def update_goal_directed_system(self,t,U):
        r_t = self.reinforcer[t-1]
        a = np.where(self.action[t-1]==1)[0]
        self.R[t] = self.R[t-1]
        self.R[t,a] = self.R[t-1,a] + self.alpha_R*(r_t - self.R[t-1,a])
        self.Q[t] = np.sum([U[i]*self.R[t,:,i] for i in range(self.nm)], axis = 0)

    def update_arbiter(self,t):
        if t == 0:
            self.gs[t] = 0
        else:
            self.gs[t] = np.sqrt(sum(self.P[t-1,i]*(self.R[t,i,1] \
                            - sum(self.P[t-1,j]*self.R[t,j,1] for j in range(self.na)))**2 for i in range(self.na)))
        
        #overall habitization
        self.hs[t] = np.sqrt(sum((self.H[t,i] - np.mean(self.H[t,:]))**2 for i in range(self.na)))
    
        #weight of habitual system
        self.weights[t] = 1/(1 + np.exp(self.w_g * self.gs[t] - self.w_h * self.hs[t] + self.w_0))
    
        #overall drive
        w = self.weights[t]
        self.D[t] = w * self.theta_h * self.H[t] + (1-w) * self.theta_g * self.Q[t]
        
    def update_pi_a(self,t):
        self.P[t] = np.array([np.exp(self.D[t,i])/sum(np.exp(self.D[t,:])) for i in range(self.na)])

    def action_selection(self,t):
        action = np.random.choice(range(self.na), p = self.P[t])
        self.action[t,action] = 1 
        return action


#%%

class sim_Free_Operant_minute(object):
    def __init__(self, trials, environment, 
                 actions_list=np.arange(0,150,1),
                 nm=3, na=2, ni_R=4, ni_H = 30,
                 alpha_H=10**(-5), alpha_R=10**(-1), 
                 w_g=6, w_h=15, w_0=1, theta_g=20, theta_h=10**3): 
        
        self.env = environment         
        
        self.trials = trials
        self.nm = nm
        self.na = na
        self.actions_list = actions_list
        self.n_action_rate = len(actions_list)
        
        self.H = np.zeros((trials,self.n_action_rate)) 
        self.R = np.zeros((trials,self.n_action_rate,nm)) 
        self.Q = np.zeros((trials,self.n_action_rate)) 
        self.D = np.zeros((trials,self.n_action_rate)) 
        self.P = np.zeros((trials,self.n_action_rate)) 
        
        self.m_a  = np.array([np.exp(-i/5) for i in self.actions_list]) # action density
        self.b_t = np.zeros((trials,nm,ni_R))
        self.c_t = np.zeros((trials,ni_H))
        self.gs = np.zeros((trials))
        self.hs = np.zeros((trials)) 
        self.weights = np.zeros((trials))

        self.action_rate  = np.zeros(trials)
        self.rate_effort_true = 2*(10**(-3))*actions_list + 6*(10**(-4))*(actions_list)**2 # for effort 
        self.reinforcers = np.zeros((trials,nm-1))
        self.reinforcers_rate = np.zeros((trials,nm-1)) # the rates of getting pellet or non-rewards of each trial
        self.effort_rate = np.zeros(trials) # the rates of effort of each trial(=action)


        self.alpha_H = alpha_H
        self.alpha_R = alpha_R        
        self.ni_R = ni_R
        self.ni_H = ni_H
        self.w_g = w_g
        self.w_h = w_h
        self.w_0 = w_0
        self.theta_h = theta_h
        self.theta_g = theta_g
        self.free_parameters = {} 
        
        self.phi = np.zeros((ni_R,self.n_action_rate))
        for i in range(ni_R):
            for j,k in enumerate(self.actions_list):
                self.phi[i,j] = ((k-75)/75)**i


        self.chi = np.zeros((ni_H,self.n_action_rate))
        for i in range(ni_H):
            for j,k in enumerate(self.actions_list):
                self.chi[i,j] = np.exp( -((k-5*i)**2)/(2*5**2))
        
        
    def run_agent(self, U, current_trial, VR = None, VI = None):
        

        self.free_parameters.update({'VR' : VR,'VI' : VI})
        
        t = current_trial 
 
        if t == 0:
            self.set_initial()
        else:
            self.update_habitual_system(t) # H
        
            self.update_goal_directed_system(t,U)  # R,Q
        
        self.update_arbiter(t) # g,h,w,D
        
        self.update_pi_a(t) #actions rate selection probability
        
        self.actioin_rate_selection(t) #select actions rate according to the probability
    
#        reinforcer,number_of_reinforcer = self.obtained_reinforcement(t,self.action_rate[t],VR,VI) #obtained reinforcer
#        self.reinforcers[t,reinforcer] = number_of_reinforcer
        
        self.obtained_reinforcement(t,self.action_rate[t],VR,VI)
        self.reinforcers_rate[t] = self.reinforcers[t]
        
        self.effort_rate[t] = self.obtained_effort(t,self.action_rate[t])
        
#        self.obtained_rewrads(t,U)
        
      
    def set_initial(self):
        self.H = np.zeros((self.trials,self.n_action_rate)) 
        self.R = np.zeros((self.trials,self.n_action_rate,self.nm)) 
        self.Q = np.zeros((self.trials,self.n_action_rate)) 

    def update_habitual_system(self,t):
        tau = np.where(self.actions_list == self.action_rate[t-1])[0] #the position of the selected actions rate
        self.c_t[t] = self.c_t[t-1] + self.alpha_H * ( 1 - self.H[t-1,tau]) * self.chi[:,tau].T \
                                    + self.alpha_H * np.sum(( 0 - self.H[t-1]) * self.chi,axis=1) \
                                    - self.alpha_H * ( 0 - self.H[t-1,tau]) * self.chi[:,tau].T
        self.H[t] = np.dot(self.c_t[t],self.chi).T
        #c_t:(trials,ni_H) ; chi(ni_H,n_action_rate)
            
    def update_goal_directed_system(self,t,U):
        tau = np.where(self.actions_list == self.action_rate[t-1])[0] #the position of the selected actions rate
        
        self.R[t] = self.R[t-1]                
        r_t = [self.reinforcers_rate[t-1,0], self.reinforcers_rate[t-1,1], self.effort_rate[t-1]]
        self.b_t[t] = self.b_t[t-1] + np.array(self.alpha_R * (r_t - self.R[t-1,tau]) * self.phi[:,tau]).T

#        for i in range(self.ni_R):
#            for k in range(self.nm):
#                self.b_t[t,k,i] += self.alpha_R*(r_t[k] - self.R[t-1,tau,k])*self.phi[i,tau]
        self.R[t] = np.dot(self.b_t[t,:],self.phi).T
        self.Q[t] = np.dot(self.R[t], U)
                
    def update_arbiter(self,t):

        if t == 0:
            self.gs[t] = 0
        else:
            self.gs[t] = np.sqrt(np.sum(self.P[t-1]*(self.R[t,:,1] - np.sum(self.P[t-1]*self.R[t,:,1]))**2))

        self.hs[t] = np.sqrt(np.sum((self.H[t] - np.mean(self.H[t]))**2))
            
        ### update weight
        self.weights[t] = 1/(1 + np.exp(self.w_g * self.gs[t] - \
                                        self.w_h * self.hs[t] + self.w_0))

        w = self.weights[t]
        
        ### update overall drive
        self.D[t] = w * self.theta_h * self.H[t] + (1-w) * self.theta_g * self.Q[t]
            
    def update_pi_a(self,t):  
        
        ### update pi(a)
        self.P[t,:] = np.exp(self.m_a*self.D[t]) / np.sum(np.exp(self.m_a*self.D[t]))

    def actioin_rate_selection(self,t):     
        ### actions rate selection    
        self.action_rate[t] = self.actions_list[np.random.choice(range(self.n_action_rate), p = self.P[t])]
    
    def obtained_reinforcement(self,t,action,VR,VI):
        
        if VR != None: # for VR schedule
            number_of_reinforcer = sum([np.random.choice(range(self.nm-1), p = [1-VR/100, VR/100]) for i in range(int(action))])
            reinforcer = 1
        else: # for VI schedule
            if action[0] == 1: # if not press, reinforcer = non-rewards
                reinforcer = 0
            elif max(self.reinforcers[1]) == 0 : # if press and haven't got pellets , get 'pellets'
                reinforcer = 1
            elif t - np.where(self.reinforcers[1] == 1)[0][-1] >= VI: # if press but already got 'pellets' in previous trial, and previous six trial did not get 'pellets', get 'pellets'
                reinforcer = 1
            else:
                reinforcer = 0
            number_of_reinforcer = 1
            
        self.reinforcers[t,reinforcer] = number_of_reinforcer
        
        return reinforcer,number_of_reinforcer
    
    def obtained_effort(self,t,action_rate):            
        ### obtained effort                        
        effort_rate = 2*(10**(-3))*action_rate + 6*(10**(-4))*action_rate**2
        self.effort_rate[t] = effort_rate
        
        return effort_rate
    
    def obtained_rewrads(self,t,U):
        ### obtained rewards
        self.rewards[t] = U[1]*self.reinforcers[t,1] + U[2]*self.effort_rate[t]       
        
        


        
