from AbstractAgent import Agent
from DQN import DQN, RND
from replay_buffer import ReplayMemory
import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque

class DQNAgent(Agent):
    
    def __init__(self, env, discount_factor = 0.99, capacity = 10000, heuristic_reward = False, RND_reward = False):
        
        if heuristic_reward and RND_reward:
            raise ValueError("can't use both : heuristic reward function and RND reward")
        
        super().__init__(env)
        self.observations = []

        # DQN classics
        self.discount_factor = discount_factor
        self.targetNet = DQN()
        self.fastupdateNet = DQN()

        # replay buffer
        self.Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))
        self.replay_buffer = ReplayMemory(capacity, self.Transition)
        self.capacity = capacity
        
        # heuristic reward function :
        self.use_heuristic_reward_function = heuristic_reward

        # random network distillation :
        self.use_RND_reward = RND_reward
        if RND_reward:
            self.RNDtargetNet = RND() # consider same structure of neural net between target and predictor. (later can be changed)
            self.RNDpredictorNet = RND()
              
    def select_action(self, state, iteration_number, starting_epsilon = 0.8, ending_epsilon = 0.05, epsilon_decay = 150):
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

            if len([s is not None for s in batch.next_state])==0 or len([s is not None for s in batch.next_state])==0 :
                break
            
            non_final_mask = torch.tensor([s is not None for s in batch.next_state], dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
            
            states = torch.cat(batch.state)
            actions = torch.cat(batch.action)
            rewards = torch.cat(batch.reward)
            
            real_batch_size = states.shape[0] #in the case we can't have a full batch of size "batch_size"
            
            state_action_values = self.fastupdateNet(states).gather(1, actions)
            next_state_values = torch.zeros(real_batch_size)
            
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

    def updateRandomNetworkDistillation(self, batch_size, learning_rate):
        list_transitions = self.replay_buffer_RND_update.sample(batch_size)
        optimizer = optim.AdamW(self.RNDpredictorNet.parameters(), lr=learning_rate, amsgrad=True)           
        for i in range(len(list_transitions)):
            transitions = list_transitions[i]
            batch = self.replay_buffer.Transition(*zip(*transitions))
            states = torch.cat(batch.state)
                
            expected_state_action_values = self.RNDtargetNet(states)
            predicted_state_action_values = self.RNDpredictorNet(states)

            criterion = nn.MSELoss()
            loss = criterion(predicted_state_action_values, expected_state_action_values)
            
            optimizer.zero_grad()
                
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.RNDpredictorNet.parameters(), 100)
            optimizer.step()
     
    def run(self, number_episode, batch_size, learning_rate):

        iteration_number = 0
        
        # Memory of all transitions during training
        if self.use_RND_reward:
            total_replay_buffer_for_RND = ReplayMemory(self.capacity, self.Transition)
        
        for i in range(number_episode):
            self.observations = []
            iteration_number += 1
            abc=0
            #creating memory for the update
            self.replay_buffer = ReplayMemory(self.capacity, self.Transition)
            
            if self.use_RND_reward :
                self.replay_buffer_RND_update = ReplayMemory(self.capacity, self.Transition)
                
            state, initial_observation = self.env.reset()
            
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            best_next_action = 100 #initialize best action for initial state (no heurisitic reward function for first action)
            
            done = False
            while not done :
                
                action = self.select_action(state, iteration_number)
                next_state, reward, terminated, truncated, _ = self.env.step(action.item())
                
                self.observations.append(next_state)
                
                if self.use_heuristic_reward_function :
                    testing_state = state.numpy()[0]
                    
                    if action == best_next_action: 
                        reward = self.heuristic_reward_function(testing_state)
                    
                    # predict the best action for next step (decide if we want to give a heuristic reward at next step)       
                    if next_state[1] < 0:
                        best_next_action = 0
                    else :
                        best_next_action = 2

                done = terminated or truncated
                
                if done :
                    next_state = None
                else :
                    next_state = torch.tensor(next_state, dtype = torch.float32).unsqueeze(0)

                state_for_RND = state
                next_state_for_RND = next_state
                
                if self.use_RND_reward and abc>1 :
                    #computing mean/std of states over all transitions during training :
                    transitions = list(total_replay_buffer_for_RND.memory)
                    past_data = total_replay_buffer_for_RND.Transition(*zip(*transitions))
                    past_states = torch.stack(past_data.state) #extract state from the memory of transitions + convert list of tensor to a single tensor
                    mean_states = past_states.mean(dim=0)
                    
                    std_state = past_states.std(dim=0)
                    
                    state_for_RND = (state - mean_states)/std_state
                    
                    if next_state == None :
                        next_state_for_RND = None
                    else :
                        next_state_for_RND = (next_state - mean_states)/std_state

                    if iteration_number > 5:
                        reward_from_MSE_nn = ((self.RNDpredictorNet(state_for_RND)-self.RNDtargetNet(state_for_RND)).item())**2

                        transitions = list(self.replay_buffer_RND_update.memory)
                        past_data = self.replay_buffer_RND_update.Transition(*zip(*transitions))
                        past_reward = torch.stack((*past_data.reward,torch.tensor([reward_from_MSE_nn])))
                        mean_reward = past_reward.mean(dim=0)
                        std_reward = past_reward.std(dim=0)
                        
                        reward = (reward_from_MSE_nn-mean_reward)/std_reward
                
                reward = torch.tensor([reward],dtype = torch.float32)
                               
                self.replay_buffer.push(state, action, next_state, reward)
                
                if self.use_RND_reward:
                    total_replay_buffer_for_RND.push(state, action, next_state, reward)
                    self.replay_buffer_RND_update.push(state_for_RND, action, next_state_for_RND, reward)

                state = next_state
                abc+=1
            # Updating neural net
            if self.use_RND_reward:
                self.updateRandomNetworkDistillation(batch_size, learning_rate)
                
            self.update(batch_size, learning_rate)
            
    def heuristic_reward_function(self, state):
        #give a reward if the action is "optimal" (in our opinion)
        position = state[0]
        speed = state[1]
        
        #form env. documentation :
        length_intervall_speed = 0.07 + 0.07
        length_intervall_position = 0.6 + 1.2
        mean_speed = 0
        mean_position = -0.5 #lowest position is approx at this position
               
        if position > -0.5:
            reward1 = abs((position-mean_position)/length_intervall_position)*1.1
        else :
            reward1 = abs((position-mean_position)/length_intervall_position)*0.7
        
        reward2 = abs((speed-mean_speed)/length_intervall_speed)
        
        reward=max(reward1,reward2)
        
        """
        if position > -0.5 :
            if speed > 0:
                reward = abs((position-mean_position)/length_intervall_position)*10 # to be define relatively to the position 
            else :
                reward = abs((speed-mean_speed)/length_intervall_speed)*10 # to be define relatively to the speed
        else : 
            if speed > 0:
                reward = abs((speed-mean_speed)/length_intervall_speed)*10 # to be define relatively to the speed
            else :
                reward = abs((position-mean_position)/length_intervall_position)*10 # te be define relatively to the position
        """        
        return reward