from AbstractAgent import Agent
import random
import numpy as np

class RandomAgent(Agent):
    
    def __init__(self, env, seed = random.randint(0,100000)):
        self.action_space = env.action_space
        self.env = env

        self.starting_state, _ = env.reset(seed = seed)

        #to store observations
        self.observed_states_speed = np.zeros((200))
        self.observed_states_position = np.zeros((200))
        self.observed_reward = np.zeros((200))
        self.nb_step = 0
        
        
    def select_action(self, state):
        #select a action randomly
        return self.action_space.sample()
        
    def observe(self, state, action, next_state, reward):
        #store observation
        self.observed_states_speed[self.nb_step] = state[1]
        self.observed_states_position[self.nb_step] = state[0]
        self.observed_reward[self.nb_step] = reward
        
    def update(self):
        #no leanring : all random
        pass

    def run(self):
        state = self.starting_state
        done = False
        #do a loop if epsiode is not finished (truncated or terminated)
        while not done :
            #select action 
            action = self.select_action(state)
            #do the step relating to action chosen
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            #store obervation
            self.observe(state, action, next_state, reward)
            # prepare next step
            state = next_state
            done = terminated or truncated
            self.nb_step +=1
