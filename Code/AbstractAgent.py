import gymnasium as gym
import numpy as np

class Agent:
    
    def __init__(self,env):
        self.starting_state, initial_observation = env.reset()
        self.action_space = env.action_space
        self.observations = []
        self.env = env
        
    def select_action(self, state):
        pass
        
    def observe(self, state, action, next_state, reward):
        pass
        
    def update(self):
        pass

    def run(self):
        pass