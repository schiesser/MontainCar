from AbstractAgent import Agent

import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim

class DynaAgent(Agent):
    
    def __init__(self, env, discr_step = [0.025, 0.005], discount_factor = 0.99, k = 1):
        super().__init__(env)

        # Environment values
        self.nb_actions = 3
        self.interval_position = [-1.2, 0.6]
        self.interval_speed = [-0.07, 0.007]

        # Initialization variables
        self.k = k
        self.discr_step = discr_step
        self.discount_factor = discount_factor

        # Calculate the number of states per interval/velocity and number of states
        self.nb_interval_position = int(abs(self.interval_position[0] - self.interval_position[1]) / discr_step[0] )
        self.nb_interval_speed = int(abs(self.interval_speed[0] - self.interval_speed[1]) / discr_step[1] )
        self.nb_states = self.nb_interval_position * self.nb_interval_speed

        # Discretization of position and speed
        self.discretization_position = np.linspace(self.interval_position[0], self.interval_position[1], self.nb_interval_position + 1)
        self.discretization_speed = np.linspace(self.interval_speed[0], self.interval_speed[1], self.nb_interval_speed + 1)

        # Tabular values :
        self.P_estimate = np.ones((self.nb_states, self.nb_actions, self.nb_states))*(1/self.nb_states) #uniform at initialization
        self.R_estimate = np.zeros((self.nb_states, self.nb_actions))
        self.Q = np.zeros((self.nb_states, self.nb_actions))
        
        
    def found_indice_bin(self, state):
        
        # Extract position and speed form state
        position = state[0]
        speed = state[1]

        # Get the indexes
        indice_position = np.digitize(position, self.discretization_position) - 1
        indice_speed = np.digitize(speed, self.discretization_speed)-1

        # Get the global index of the state
        return indice_position * self.nb_interval_speed + indice_speed
        

    def found_state(self, correct_indice): #not sure it will be usefull, do the inverse of found_indice_bin : get position/speed interval from and indice
        
        indice_speed = correct_indice % self.nb_interval_speed
        indice_position = int((correct_indice - indice_speed)/self.nb_interval_speed)

        position = np.array([self.discretization_position[indice_position], self.discretization_position[indice_position]+1])
        speed = np.array([self.discretization_speed[indice_speed], self.discretization_speed[indice_speed]+1])

        return position,speed


    def select_action(self, state, iteration_number, starting_epsilon = 0.8, ending_epsilon = 0.05, epsilon_decay = 150):
        """
        Chooses an epsilon-greedy action given a state and its Q-values associated
        parameters : starting_epsilon, ending_epsilon, epsilon_decay allow to manage exploitation and exploration during action selection
        epsilon is decresing relatively to the number of iterations
        """
        # reason : more exploration at beginning, and increase exploitation with iteration
        epsilon = ending_epsilon + (starting_epsilon - ending_epsilon) * math.exp(-iteration_number/epsilon_decay)

        indice_state = self.found_indice_bin(state) 
        
        if np.random.uniform(0, 1) > epsilon:
            best_action = np.max(self.Q[indice_state,:])
            return best_action
        else :
            random_action = np.random.choice(self.Q[indice_state,:])# choose a random action
            return random_action
    
    
    def run():
        pass
                