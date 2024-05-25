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
from tqdm import tqdm

class DQNAgent(Agent):
    
    def __init__(self, env, discount_factor = 0.99, starting_epsilon = 0.8, ending_epsilon = 0.05, epsilon_decay = 150, capacity = 10000, heuristic_reward = False, RND_reward = False, factor_favorize_reward = 1, global_reward_factor = 1, criterion = "MSE", neurons_RND = 16, use_log = False,lr_RND=0.05 ):#paramter to load or not the past trained weight of the neural.
        if heuristic_reward and RND_reward:
            raise ValueError("can't use both : heuristic reward function and RND reward")
        self.lr_RND=lr_RND
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
        
        if self.use_RND_reward:
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
        self.use_log = use_log
        
        if self.use_log :    
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

    def updateRandomNetworkDistillation(self, mean_states,std_states, learning_rate, batch_size):

        list_transitions = self.replay_buffer.sample(batch_size)

        #loop over 5 lists of transitions for a batch update
        for i in range(min(len(list_transitions),20)): 
            transitions = list_transitions[i] 
            #reorganize the storage of the transitions 
            batch = self.replay_buffer.Transition(*zip(*transitions)) 
            
            states = torch.cat(batch.state)

            normalized_states = (states-mean_states)/std_states
            #compute output of both vector
            expected_value = self.RNDtargetNet(normalized_states)
            predicted_value = self.RNDpredictorNet(normalized_states)
    
            # compute the loss using mean square error
            # loss is : how different are the 2 neural net for the given state
            loss = self.criterionDQN(predicted_value, expected_value)
    
            # optimize parameter of RNDpredictor
            # goal : to be close to the other neural for the actions that have already been observed !
            optimizer = optim.AdamW(self.RNDpredictorNet.parameters(), lr=self.lr_RND)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.RNDpredictorNet.parameters(), 100) # clip values of parameter between -100 and 100 : prevent exploding gradient
            optimizer.step()
     
    def run(self, number_episode, batch_size, learning_rate):
        past = False
        #iteration number (episode)
        iteration_number = 0
        #number of past steps during training
        nb_transition_tot = 0
        # use for first cumputation of mean/std :
        mean_std_to_compute = True

        #track cumulative task solved during training :
        tasksolve = 0
        #count env. reward per episode :
        environnement_reward = np.zeros((number_episode))
        #count total reward per episode :
        total_reward = np.zeros((number_episode))

        #creating memory for the update of DQN
        self.replay_buffer = ReplayMemory(self.capacity, self.Transition)
          
        for i in tqdm(range(number_episode)):
            #variable to count env reward per ep.:
            environnement_reward_ep = 0
            #variable to count total reward per ep. :
            total_reward_ep = 0

            #to visualize states of last episode :
            self.observations = []

            #count iteration number :
            iteration_number += 1
            #count nbr of transition per ep. :
            nb_trans_ep = 0
                 
            #initialisation of episode :
            state, initial_observation = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            best_next_action = 100 #initialize best action for initial state (no heurisitic reward function for first action)
            done = False #to stop episode
            while not done :

                nb_trans_ep +=1
                nb_transition_tot += 1
                
                action = self.select_action(state, iteration_number)
                next_state, reward, terminated, truncated, _ = self.env.step(action.item())
                done = terminated or truncated
                
                environnement_reward_ep += reward
                self.observations.append(next_state)
                
                # coumpte heuristic reward if chosen
                if self.use_heuristic_reward_function :
                    testing_state = state.numpy()[0]
                    
                    if action == best_next_action:
                        reward += self.heuristic_reward_function(testing_state)*10
                    
                    # predict the best action for next step (decide if we want to give a heuristic reward at next step)       
                    if next_state[1] < 0:
                        best_next_action = 0
                    else :
                        best_next_action = 2

                #deal with last step : (do sth special if done...)
                if done :
                    next_state = None
                else :
                    next_state = torch.tensor(next_state, dtype = torch.float32).unsqueeze(0)

                # compute RND reward if chosen
                if self.use_RND_reward and (i>27 or nb_transition_tot > 5000):

                    if mean_std_to_compute and nb_transition_tot == 5001 :#first computation of mean after 5 transitions in the first episode
                        print("initilization")
                        mean_std_to_compute = False #to not enter again in if statement

                        #compute mean/std states with past 5 transitions
                        transitions = list(self.replay_buffer.memory)
                        past_data = self.replay_buffer.Transition(*zip(*transitions))
                        past_states = torch.stack(past_data.state) #extract state from the memory of transitions + convert list of tensor to a single tensor
                        mean_states = past_states.mean(dim=0)
                        std_states = past_states.std(dim=0)
                        past_reward = torch.stack(past_data.reward)

                        #normalizing states
                        normalized_state = (state-mean_states)/(std_states+10e-8)
                        normalized_past_states = (past_states - mean_states) / (std_states+10e-8)

                        # compute std and means for RND rewards
                        past_RND_rewards = (self.RNDtargetNet(normalized_past_states) - self.RNDpredictorNet(normalized_past_states))**2
                        #print("used to compute RND reward mean/std ")
                        #print(past_RND_rewards)
                        #print("all past RND reward")
                        #print(past_RND_rewards)
                        mean_RND_rewards = past_RND_rewards.mean(dim=0)
                        std_RND_rewards = past_RND_rewards.std(dim=0)
                        #print("mean_RND_rewards")
                        #print(mean_RND_rewards)
                        #print("std_RND_rewards")
                        #print(std_RND_rewards)

                        #compute current RND reward 
                        not_normalized_RND_reward = (self.RNDtargetNet(normalized_state) - self.RNDpredictorNet(normalized_state))**2
                        RND_reward = (not_normalized_RND_reward-mean_RND_rewards)/(std_RND_rewards+10e-8)
                        #print("first not_normalized_RND_reward")
                        #print(not_normalized_RND_reward)
                        #print("first RND reward")
                        #print(RND_reward)

                    #normalize current state and next_state :
                    normalized_state = (state-mean_states)/std_states
                    #print("First normalizeed state")
                    #print(normalized_state)
                    
                    if next_state is None :
                        normalized_next_state = None
                    else :
                        normalized_next_state = (next_state - mean_states)/(std_states+10e-8)
                        
                        #updating the mean/std of past states :
                        mean_states = ((mean_states*nb_transition_tot) + state)/(nb_transition_tot+1)

                        d_s = state - mean_states
                        std_states = ((nb_transition_tot*std_states**2+d_s**2)/(nb_transition_tot+1))**(0.5)
                        
                        #compute RND reward
                        not_normalized_RND_reward = (self.RNDpredictorNet(normalized_next_state)-self.RNDtargetNet(normalized_next_state))**2

                        mean_RND_rewards = ((mean_RND_rewards*nb_transition_tot) + not_normalized_RND_reward)/(nb_transition_tot+1)
                        d_r = not_normalized_RND_reward - mean_RND_rewards
                        std_RND_rewards = ((nb_transition_tot*std_RND_rewards**2+d_r**2)/(nb_transition_tot+1))**(0.5)
                        
                        RND_reward = (not_normalized_RND_reward-mean_RND_rewards)/(std_RND_rewards+10e-8)
                        #print("not_normalized_RND_reward")
                        #print(not_normalized_RND_reward)
                        #print("mean_RND_rewards")
                        #print(mean_RND_rewards)
                        #print("std_RND_rewards")
                        #print(std_RND_rewards)
                        #print("RND_reward")
                        #print(RND_reward)
                        #updating the mean/std of past rewards :
                        
                        clamped_RND_reward = torch.clamp(RND_reward, min=-5, max= 5).item()
                        reward = reward + abs(clamped_RND_reward)*self.global_reward_factor
                    
                total_reward_ep += reward
                
                reward = torch.tensor([reward],dtype = torch.float32)
                
                self.replay_buffer.push(state, action, next_state, reward)

                state = next_state
                environnement_reward[i] = environnement_reward_ep
                total_reward[i] = total_reward_ep
            
            # Updating RND neural net with next_state normalized
            if (not mean_std_to_compute) and (self.use_RND_reward) and normalized_next_state is not None:
                #print("correctly updated")
                self.updateRandomNetworkDistillation(mean_states,std_states, learning_rate, batch_size)
                
            self.update(batch_size, learning_rate)

            if nb_trans_ep < 200:
                tasksolve+=1 
            
            if self.use_log : 
                self.writer.add_scalar('Reward/Episode', self.loss, i)
                self.writer.add_scalar('Reward/Episode', reward.item(), i)
                self.writer.add_scalar('Nb_steps/Episode', nb_trans_ep, i)
                self.writer.add_scalar('environnement_reward/Episode', environnement_reward_ep, i)
                self.writer.add_scalar('total_reward/Episode', total_reward_ep, i)
                self.writer.add_scalar('auxiliary_reward/Episode', total_reward_ep-environnement_reward_ep, i)  
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