from AbstractAgent import Agent
from DQN import DQN
from replay_buffer import ReplayMemory
import math
import random
import torch
import torch.nn as nn
import torch.optim as optim

class DQNAgent(Agent):
    
    def __init__(self, env, discount_factor = 0.99, capacity = 10000):
        super().__init__(env)
        self.discount_factor = discount_factor
        self.observations = []
        self.targetNet = DQN()
        self.fastupdateNet = DQN()
        self.replay_buffer = ReplayMemory(capacity)
        self.capacity = capacity
        
    def select_action(self, state, iteration_number, starting_epsilon, ending_epsilon, epsilon_decay):
        """
        Chooses an epsilon-greedy action given a state and its Q-values associated
        parameters : starting_epsilon, ending_epsilon, epsilon_decay allow to manage exploitation and exploration during action selection
        epsilon is decresing relatively to the number of iterations
        """
        epsilon = ending_epsilon + (starting_epsilon - ending_epsilon) * math.exp(-iteration_number/epsilon_decay) #reason : more exploration at beginning, and increase exploitation with iteration
        
        if np.random.uniform(0, 1) > epsilon:
            with torch.no_grad(): #doesn't track the operation for the training
                return policy_net(state).max(1).indices.view(1, 1)
        else :
            return torch.tensor([[self.action_space.sample()]], device=device, dtype=torch.long)
        
    def update(self, batch_size, learning_rate):

        list_transitions = self.replay_buffer.memory.sample(batch_size)
        
        for i in range(len(list_transitions)):
            transitions = list_transitions[i]
            batch = Transition(*zip(*transitions))

            non_final_mask = torch.tensor(tuple(map(lambda s:s is not None, batch.next_state)), dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state])

            states = torch.cat([batch.state])
            actions = torch.cat([batch.state])
            rewards = torch.cat([batch.state])

            state_action_values = self.fastupdateNet(state_batch).gather(1, action_batch)
            next_state_values = torch.zeros(BATCH_SIZE, device=device)
            
            with torch.no_grad():
                next_state_values[non_final_mask] = self.targetNet(non_final_next_states).max(1).values

            expected_state_action_values = (next_state_values * self.discount_factor) + reward_batch

            criterion = nn.MSELoss()
            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

            optimizer = optim.AdamW(self.fastupdateNet.parameters(), lr = learning_rate, amsgrad=True)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.fastupdateNet.parameters(), 100) # don't understand the reason of the usage
            optimizer.step()
        self.targetNet.state_dict()=self.fastupdateNet.state_dict()

    def run(self, number_episode):

        for i in range(number_episode):
            
            self.replay_buffer = ReplayMemory(self.capacity)
            
            state, initial_observation = env.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            done = False
            while not done :
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action.item())
                reward = torch.tensor([reward])
                done = terminated or truncated
                
                if done :
                    next_state = None
                else :
                    next_state = torch.tensor(next_state, dtype = torch.float32).unsqueeze(0)
                
                self.replay_buffer.push(state, action, next_state, reward)

                state = next_state

        
        
    def 
        
