from AbstractAgent import Agent
from DQN import DQN
from replay_buffer import ReplayMemory
import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque

class DQNAgent(Agent):
    
    def __init__(self, env, discount_factor = 0.99, capacity = 10000, heuristic_reward = False):
        super().__init__(env)
        self.Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))
        self.discount_factor = discount_factor
        self.observations = []
        self.targetNet = DQN()
        self.fastupdateNet = DQN()
        self.replay_buffer = ReplayMemory(capacity, self.Transition)
        self.capacity = capacity
        self.use_heuristic_reward_function = heuristic_reward
        
    def select_action(self, state, iteration_number, starting_epsilon = 0.9, ending_epsilon = 0.05, epsilon_decay = 1000):
        """
        Chooses an epsilon-greedy action given a state and its Q-values associated
        parameters : starting_epsilon, ending_epsilon, epsilon_decay allow to manage exploitation and exploration during action selection
        epsilon is decresing relatively to the number of iterations
        """
        epsilon = ending_epsilon + (starting_epsilon - ending_epsilon) * math.exp(-iteration_number/epsilon_decay) #reason : more exploration at beginning, and increase exploitation with iteration
        
        if np.random.uniform(0, 1) > epsilon:
            with torch.no_grad(): #doesn't track the operation for the training
                return self.fastupdateNet(state).max(1).indices.view(1, 1)
        else :
            return torch.tensor([[self.action_space.sample()]], dtype=torch.long)
        
    def update(self, batch_size, learning_rate):

        list_transitions = self.replay_buffer.sample(batch_size)
                
        for i in range(len(list_transitions)):
            transitions = list_transitions[i]
            batch = self.replay_buffer.Transition(*zip(*transitions))

            non_final_mask = torch.tensor(tuple(map(lambda s:s is not None, batch.next_state)), dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

            states = torch.cat(batch.state)
            actions = torch.cat(batch.action)
            rewards = torch.cat(batch.reward)

            state_action_values = self.fastupdateNet(states).gather(1, actions)
            next_state_values = torch.zeros(batch_size)
            
            with torch.no_grad():
                next_state_values[non_final_mask] = self.targetNet(non_final_next_states).max(1).values

            expected_state_action_values = (next_state_values * self.discount_factor) + rewards

            criterion = nn.MSELoss()
            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

            optimizer = optim.AdamW(self.fastupdateNet.parameters(), lr = learning_rate, amsgrad=True)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.fastupdateNet.parameters(), 100) # don't understand the reason of the usage
            optimizer.step()
            
        self.targetNet.load_state_dict(self.fastupdateNet.state_dict())

    def run(self, number_episode, batch_size, learning_rate):

        iteration_number = 0
        
        for i in range(number_episode):
            
            iteration_number += 1
            
            self.replay_buffer = ReplayMemory(self.capacity, self.Transition)
            
            state, initial_observation = self.env.reset()
            
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            best_next_action = 100
            
            done = False
            while not done :
                
                action = self.select_action(state, iteration_number)
                next_state, reward, terminated, truncated, _ = self.env.step(action.item())
                
                if self.use_heuristic_reward_function :
                    testing_state = state.numpy()[0]
                    
                    if action == best_next_action: 
                        if testing_state[0] > -0.5 :
                            if testing_state[1] > 0:
                                reward = 10 # to be define relatively to the position 
                            else :
                                reward = 10 # to be define relatively to the speed
                        else : 
                            if testing_state[1] > 0:
                                reward = 10 # to be define relatively to the speed
                            else :
                                reward = 10 # te be define relatively to the position
                                
                    if next_state[1] < 0:
                        best_next_action = 0
                    else :
                        best_next_action = 2
                        
                reward = torch.tensor([reward])
                done = terminated or truncated
                
                if done :
                    next_state = None
                else :
                    next_state = torch.tensor(next_state, dtype = torch.float32).unsqueeze(0)
                
                self.replay_buffer.push(state, action, next_state, reward)

                state = next_state

            self.update(batch_size, learning_rate)
        