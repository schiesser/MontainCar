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
from torch.utils.tensorboard import SummaryWriter
import datetime

class DQNAgent(Agent):
    
    def __init__(self, env, discount_factor = 0.99, starting_epsilon = 0.8, ending_epsilon = 0.05, epsilon_decay = 150, capacity = 10000, heuristic_reward = False, RND_reward = False, factor_favorize_reward = 1, global_reward_factor = 1, criterion = "MSE", neurons_RND = 16 ):#paramter to load or not the past trained weight of the neural.
        if heuristic_reward and RND_reward:
            raise ValueError("can't use both : heuristic reward function and RND reward")
        
        super().__init__(env)
        self.observations = []

        # DQN classics
        self.targetNet = DQN()
        self.fastupdateNet = DQN()
        self.targetNet.load_state_dict(self.fastupdateNet.state_dict()) #we want to have to same initial parameters for both NN.

        # replay buffer
        self.Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))
        self.replay_buffer = ReplayMemory(capacity, self.Transition)
        self.capacity = capacity
        
        # heuristic reward function :
        self.use_heuristic_reward_function = heuristic_reward

        # random network distillation :
        self.use_RND_reward = RND_reward
        if RND_reward:
            # the 2 networks have different parameters at initilization ! (At moment i let default initilisation)
            self.RNDtargetNet = RND(nb_neurons = neurons_RND) 
            self.RNDpredictorNet = RND(nb_neurons = neurons_RND)

        # others hyperparameters :
        self.discount_factor = discount_factor
        self.epsilon_decay = epsilon_decay
        
        if criterion == "MSE":
            self.criterionDQN = nn.MSELoss()
        elif criterion == "L1":
            self.criterionDQN = nn.L1Loss()
        else :
            raise ValueError("unknown criterion")
        
        self.factor_favorize_reward = factor_favorize_reward
        self.global_reward_factor = global_reward_factor
        self.ending_epsilon = ending_epsilon
        self.starting_epsilon = starting_epsilon
            
        if heuristic_reward :# Writer for logging purpose
            logdir = f'./DQN/@e_decay={str(epsilon_decay)}@disc_factor={str(discount_factor)}@global_reward={str(global_reward_factor)}@crit={str(criterion)}@{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
        elif RND_reward :
            logdir = f'./DQN_RND/@e_decay={str(epsilon_decay)}@disc_factor={str(discount_factor)}@global_reward={str(global_reward_factor)}@crit={str(criterion)}@{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
        self.writer = SummaryWriter(log_dir=logdir)
        print(f"------------------------------------------\nWe will log this experiment in directory {logdir}\n------------------------------------------")
              
    def select_action(self, state, iteration_number):
        """
        Chooses an epsilon-greedy action given a state and its Q-values associated
        parameters : starting_epsilon, ending_epsilon, epsilon_decay allow to manage exploitation and exploration during action selection
        epsilon is decresing relatively to the number of iterations
        """
        # reason : more exploration at beginning, and increase exploitation with iteration
        epsilon = self.ending_epsilon + (self.starting_epsilon - self.ending_epsilon) * math.exp(-iteration_number/self.epsilon_decay)
        
        if np.random.uniform(0, 1) > epsilon:
            with torch.no_grad(): #doesn't track the operation for the training
                return self.fastupdateNet(state).max(1).indices.view(1, 1) # choose current best action
        else :
            return torch.tensor([[self.action_space.sample()]], dtype=torch.long) # choose a random action
        
    def update(self, batch_size, learning_rate):

        # split all transitions randomly into lists of batch_size length (except last list...)
        list_transitions = self.replay_buffer.sample(batch_size)

        #loop over 5 lists of transitions for a batch update
        for i in range(min(len(list_transitions),20)): 
            transitions = list_transitions[i] 
            # reorganize the storage of the transitions : https://stackoverflow.com/a/19343/3343043
            batch = self.replay_buffer.Transition(*zip(*transitions)) 

            # break loop if a batch is composed only of a ending state
            # possible like this because only the last batch is not of length < batch_size
            # (not sure it's working yet)
            if len(batch.next_state)==0 : 
                break
            
            #deal with ending state (which has "None" value)
            non_final_mask = torch.tensor([s is not None for s in batch.next_state], dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
            
            #extract states, actions, rewards from the batch
            states = torch.cat(batch.state)
            actions = torch.cat(batch.action)
            rewards = torch.cat(batch.reward)
            
            #for the last batch : in the case we can't have a full batch of size "batch_size"
            real_batch_size = states.shape[0] 
            
            # Q values of each (s,a) of the batch (relatively to fastupdateNet)
            state_action_values = self.fastupdateNet(states).gather(1, actions)
            
            # initialize a torch for the storage of the current optimal action (relatively to targetNet)
            next_state_values = torch.zeros(real_batch_size)
            with torch.no_grad():
                #store the current optimal action (relatively to targetNet)
                next_state_values[non_final_mask] = self.targetNet(non_final_next_states).max(1).values
            
            #from Bellman...
            expected_state_action_values = (next_state_values * self.discount_factor) + rewards

            #compute loss with mean square error criterion
            self.loss = self.criterionDQN(state_action_values, expected_state_action_values.unsqueeze(1))

            #optimize the parameters of fastupdateNet :
            optimizer = optim.AdamW(self.fastupdateNet.parameters(), lr = learning_rate, amsgrad=True)
            optimizer.zero_grad()
            self.loss.backward()
            torch.nn.utils.clip_grad_value_(self.fastupdateNet.parameters(), 100) # clip values of parameter between -100 and 100 : prevent exploding gradient
            optimizer.step()
            
        # paste the parameter of fastupdateNet that were optimize by few batchs to the targetNet
        self.targetNet.load_state_dict(self.fastupdateNet.state_dict())

    def updateRandomNetworkDistillation(self, batch_size, learning_rate):

        # split all transitions randomly into lists of batch_size length (except last list...)
        list_transitions = self.replay_buffer_RND_update.sample(batch_size)

        #loop over lists of transitions for a batch update
        for i in range(min(len(list_transitions),30)):
            transitions = list_transitions[i]
            # reorganize the storage of the transitions
            batch = self.replay_buffer_RND_update.Transition(*zip(*transitions))
            
            # extract states form the batch of transition
            states = torch.cat(batch.state)

            # compute output of both neural
            expected_state_action_values = self.RNDtargetNet(states)
            predicted_state_action_values = self.RNDpredictorNet(states)

            # compute the loss using mean square error
            # loss is : how different are the 2 neural net for these given states
            loss = self.criterionDQN(predicted_state_action_values, expected_state_action_values)

            # optimize parameter of RNDpredictor
            # goal : to be close to the other neural for the actions that have already been observed !
            optimizer = optim.AdamW(self.RNDpredictorNet.parameters(), lr=learning_rate, amsgrad=True)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.RNDpredictorNet.parameters(), 100) # clip values of parameter between -100 and 100 : prevent exploding gradient
            optimizer.step()
     
    def run(self, number_episode, batch_size, learning_rate):
        tasksolve = 0
        iteration_number = 0

        environnement_reward = np.zeros((number_episode))
        total_reward = np.zeros((number_episode))

        nb_transition_tot = 0
        abc = 0
        
        #creating memory for the update
        self.replay_buffer = ReplayMemory(self.capacity, self.Transition)
        
        if self.use_RND_reward :
                self.replay_buffer_RND_update = ReplayMemory(self.capacity, self.Transition)
            
            
        for i in range(number_episode):
            environnement_reward_ep = 0
            total_reward_ep = 0
            self.observations = []
            iteration_number += 1
            nb_trans_ep = 0
                 
            state, initial_observation = self.env.reset()
            
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            best_next_action = 100 #initialize best action for initial state (no heurisitic reward function for first action)
            
            done = False
            while not done :
                nb_trans_ep +=1
                nb_transition_tot += 1
                action = self.select_action(state, iteration_number)
                next_state, reward, terminated, truncated, _ = self.env.step(action.item())
                environnement_reward_ep += reward
                
                self.observations.append(next_state)
                
                if self.use_heuristic_reward_function :
                    testing_state = state.numpy()[0]
                    
                    if action == best_next_action:
                        reward += self.heuristic_reward_function(testing_state)*10
                    
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
                
                if self.use_RND_reward and (i>0 or nb_trans_ep > 4):
                    if i == 0 and abc == 0 and nb_trans_ep == 5 :#first computation of mean after 5 transitions in the first episode
                        abc += 1
                        transitions = list(self.replay_buffer_RND_update.memory)
                        past_data = self.replay_buffer_RND_update.Transition(*zip(*transitions))
                        past_states = torch.stack(past_data.state) #extract state from the memory of transitions + convert list of tensor to a single tensor
                        mean_states = past_states.mean(dim=0)
                        std_states = past_states.std(dim=0)
                        past_reward = torch.stack(past_data.reward)
                        mean_reward = past_reward.mean(dim=0)
                        std_reward = past_reward.std(dim=0)

                    previous_mean_states = mean_states
                    mean_states = mean_states + (state-mean_states)/(nb_transition_tot+1)
                    std_states = ((state-previous_mean_states)*(state-mean_states)+std_states)/nb_transition_tot
   
                    state_for_RND = (state - mean_states)/std_states
                    
                    if next_state == None :
                        next_state_for_RND = None
                    else :
                        next_state_for_RND = (next_state - mean_states)/std_states

                    reward_from_MSE_nn = ((self.RNDpredictorNet(state_for_RND)-self.RNDtargetNet(state_for_RND)).item())**2
                    
                    previous_mean_reward = mean_reward
                    mean_reward = mean_reward + (reward_from_MSE_nn-mean_reward)/(nb_transition_tot+1)
                    std_reward = ((reward_from_MSE_nn-previous_mean_reward)*(reward_from_MSE_nn-mean_reward)+std_reward)/nb_transition_tot  
                    rnd_reward = (reward_from_MSE_nn-mean_reward)/std_reward
                    rnd_reward = torch.clamp(rnd_reward, min=-5, max= 5).item()
                    reward = reward+abs(rnd_reward)*self.global_reward_factor
                    
                total_reward_ep += reward
                
                reward = torch.tensor([reward],dtype = torch.float32)
                
                self.replay_buffer.push(state, action, next_state, reward)
                
                if self.use_RND_reward:
                    self.replay_buffer_RND_update.push(state_for_RND, action, next_state_for_RND, reward)

                state = next_state
                environnement_reward[i] = environnement_reward_ep
                total_reward[i] = total_reward_ep
                
            # Updating neural net
            if self.use_RND_reward:
                self.updateRandomNetworkDistillation(batch_size, learning_rate)
                
            self.update(batch_size, learning_rate)
            self.writer.add_scalar('Reward/Episode', self.loss, i)
            self.writer.add_scalar('Reward/Episode', reward.item(), i)
            self.writer.add_scalar('Nb_steps/Episode', nb_trans_ep, i)
            self.writer.add_scalar('environnement_reward/Episode', environnement_reward_ep, i)
            self.writer.add_scalar('total_reward/Episode', total_reward_ep, i)
            self.writer.add_scalar('auxiliary_reward/Episode', total_reward_ep-environnement_reward_ep, i)
            if nb_trans_ep < 200:
                tasksolve+=1
                
            self.writer.add_scalar('solve_task/Episode', tasksolve, i)
            self.writer.flush()
        
        auxiliary_reward = total_reward - environnement_reward
        
        return environnement_reward, auxiliary_reward, total_reward
            
    def heuristic_reward_function(self, state):
        #give a reward if the action is "optimal" (in our opinion)

        #extract position and speed for a given state
        position = state[0]
        speed = state[1]
        
        #computing few things using the documentation of the environnement:
        length_intervall_speed = 0.07 + 0.07
        length_intervall_position = 0.6 + 1.2
        mean_speed = 0
        mean_position = -0.5 #lowest position is approx at this position
               
        # compute reward relatively to the positon and speed
        # basically a normalization is done
        # factor 1.1 and 0.7 consider that the mean position is closer to the left boundary than the right booundary(goal)
        if position > -0.5:
            reward1 = abs((position-mean_position)/length_intervall_position)*1.1*self.factor_favorize_reward #bonus factor of 1.25 : to favorize right side 
            reward2 = abs((speed-mean_speed)/length_intervall_speed)*self.factor_favorize_reward
        else :
            reward1 = abs((position-mean_position)/length_intervall_position)*0.7
            reward2 = abs((speed-mean_speed)/length_intervall_speed)

        #keep max reward
        reward=max(reward1,reward2)*self.global_reward_factor
        
        return reward