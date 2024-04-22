import gymnasium as gym
import numpy as np
env = gym.make('MountainCar-v0')

class Agent:
    
    def __init__(self):
        self.starting_state, initial_observation = env.reset()
        self.action_space = env.action_space
        self.observations = []
        

class RandomAgent(Agent):
    
    def __init__(self):
        super().__init__()
        
    def select_action(self, state):
        return self.action_space.sample()
        
    def observe(self, state, action, next_state, reward):
        current_observation = (state, action, next_state, reward)
        self.observations.append(current_observation)
        
    def update(self):
        pass

    def run(self):
        state = self.starting_state
        done = False
        episode_reward = 0
        while not done :
            action = self.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            self.observe(state, action, next_state, reward)
            
            episode_reward += reward
            state = next_state
            done = terminated or truncated
            
        print(f"Episode reward after taking random actions: {episode_reward}")
        
class DQNAgent(Agent):
    
    def __init__(self):
        super().__init__()


class DynaAgent(Agent):
    
    def __init__(self):
        super().__init__()
        