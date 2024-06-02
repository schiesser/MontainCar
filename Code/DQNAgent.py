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
import os

class DQNAgent(Agent):
    
    def __init__(self, env, discount_factor = 0.99, epsilon_decay = 150, capacity = 10000, heuristic_reward = False, RND_reward = False, global_reward_factor = 1, neurons_RND = 32, neurons_DQN = 32, use_log = False, lr_RND =0.0025, lr_DQN =0.005, load_model = False ):
        self.nb_step_training=0
        if heuristic_reward and RND_reward:
            raise ValueError("Careful you use both : heuristic reward function and RND reward")
        
        #learning rate for DQN and RND
        self.lr_RND=lr_RND
        self.lr_DQN = lr_DQN
        super().__init__(env)
        self.observations = []

        # DQN neural network
        self.targetNet = DQN(nb_neurons = neurons_DQN)
        self.fastupdateNet = DQN(nb_neurons = neurons_DQN)
        self.targetNet.load_state_dict(self.fastupdateNet.state_dict())#we want to have to same initial parameters for both NN.

        # replay buffer : tuple with name "transitions" and containing state action, next_state, reward component for each transitions.
        self.Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))
        self.replay_buffer = ReplayMemory(capacity, self.Transition)
        self.capacity = capacity
        
        ## Choose a auxiliary reward method
        # heuristic reward function :
        self.use_heuristic_reward_function = heuristic_reward
        # random network distillation :
        self.use_RND_reward = RND_reward
        
        if self.use_RND_reward:
            # initialization of RND with independent weights
            self.RNDtargetNet = RND(nb_neurons = neurons_RND) 
            self.RNDpredictorNet = RND(nb_neurons = neurons_RND)

        # Hyperparameter :
        self.discount_factor = discount_factor
        self.epsilon_decay = epsilon_decay
        
        # to scale auxiliary reward
        self.global_reward_factor = global_reward_factor

        # use tensorboard or not
        self.use_log = use_log
        
        if self.use_log :
            logdir = f'./RNDcomplast/@factor={str(global_reward_factor)}@lr_DQN={str(lr_DQN)}@cap={str(capacity)}@{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
            self.writer = SummaryWriter(log_dir=logdir)
            print(f"------------------------------------------\nWe will log this experiment in directory {logdir}\n------------------------------------------")
        
        #store running after training
        self.observed_states_speed = np.zeros((200))
        self.observed_states_position = np.zeros((200))
        self.observed_reward = np.zeros((200))

        #not necessary to retrain all....
        if load_model:
            directory = os.path.dirname(os.path.abspath(__file__))
            
            if heuristic_reward:
                load_path = os.path.join(directory, "optimizeDQN_heuristic.pth")
            elif RND_reward:
                load_path = os.path.join(directory, "optimizeDQN_RND.pth")
            else :
                load_path = os.path.join(directory, "optimizeDQN.pth")
                
            state_dict = torch.load(load_path)
            self.targetNet.load_state_dict(state_dict)
            self.fastupdateNet.load_state_dict(state_dict)
            
    def select_action(self, state, iteration_number, starting_epsilon = 0.9, ending_epsilon = 0.05):
        
        # epsilon greedy policy
        # vary during the training : initialized at 0.9 and exponentially decays until it reaches a minimum value of 0.05.
        # reason : more exploration at beginning, and increase exploitation with iterations
        
        epsilon = ending_epsilon + (starting_epsilon - ending_epsilon) * math.exp(-iteration_number/self.epsilon_decay)
        
        if np.random.uniform(0, 1) > epsilon:
            with torch.no_grad(): #doesn't track the operation for the training
                return self.fastupdateNet(state).max(1).indices.view(1, 1) # choose current best action
        else :
            return torch.tensor([[self.action_space.sample()]], dtype=torch.long) # choose a random action
        
    def update(self, batch_size, learning_rate):

        # split all transitions randomly into lists of batch_size length (except last list...)
        list_transitions = self.replay_buffer.sample(batch_size)
        
        loss_update = 0
        # Update loop over every batch list of transition (except last one cause it can be a very small list)
        for i in range(len(list_transitions)-1): 
            transitions = list_transitions[i] 
            # reorganize the storage of the transitions
            batch = self.replay_buffer.Transition(*zip(*transitions)) 
            
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
            criterion = nn.SmoothL1Loss()
            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
            
            #optimize the parameters of fastupdateNet :
            optimizer = optim.AdamW(self.fastupdateNet.parameters(), lr = learning_rate, amsgrad=True)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.fastupdateNet.parameters(), 100) # clip values of parameter between -100 and 100 : prevent exploding gradient
            optimizer.step()
            
            loss_update += loss.item()
            
        # paste the parameter of fastupdateNet that were optimize by few batchs to the targetNet
        self.targetNet.load_state_dict(self.fastupdateNet.state_dict())

        return loss
        
    def updateRandomNetworkDistillation(self, mean_states,std_states, batch_size):

        list_transitions = self.replay_buffer.sample(batch_size)
        
        loss_update = 0
        for i in range(len(list_transitions)-1): 
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
            criterion = nn.MSELoss()
            loss = criterion(predicted_value, expected_value)
            
            # optimize parameter of RNDpredictor
            # goal : to be close to the other neural for the actions that have already been observed !
            optimizer = optim.AdamW(self.RNDpredictorNet.parameters(), lr=self.lr_RND)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.RNDpredictorNet.parameters(), 100) # clip values of parameter between -100 and 100 : prevent exploding gradient
            optimizer.step()

            loss_update += loss.item()

        return loss_update
            
    def observe_learning(self, state, action, next_state, reward):
        self.replay_buffer.push(state, action, next_state, reward)

    def normalize_state(self, next_state, mean_states, std_states):
                 
        if next_state is None :
            normalized_next_state = None
        else :
            normalized_next_state = (next_state - mean_states)/(std_states+10e-8)

        return normalized_next_state
    
    def update_mean_std_states(self, next_state, mean_states, std_states, nb_transition_tot):
        
        mean_states = ((mean_states*nb_transition_tot) + next_state)/(nb_transition_tot+1)
        d_s = next_state - mean_states
        std_states = ((nb_transition_tot*std_states**2+d_s**2)/(nb_transition_tot+1))**(0.5)

        return mean_states, std_states

    def update_mean_std_aux_reward(self, new_reward, mean_RND_rewards, std_RND_rewards, nb_transition_tot):

        mean_RND_rewards = ((mean_RND_rewards*nb_transition_tot) + new_reward)/(nb_transition_tot+1)
        d_r = new_reward - mean_RND_rewards
        std_RND_rewards = ((nb_transition_tot*std_RND_rewards**2+d_r**2)/(nb_transition_tot+1))**(0.5)

        return mean_RND_rewards, std_RND_rewards
    
    def normalize_RND_reward(self, new_reward, mean_RND_rewards, std_RND_rewards):
        
        RND_reward = (new_reward-mean_RND_rewards)/(std_RND_rewards+10e-8)

        return RND_reward

    def init_mean_std(self):
        
        transitions = list(self.replay_buffer.memory)
        past_data = self.replay_buffer.Transition(*zip(*transitions))
        past_states = torch.stack(past_data.state)
        mean_states = past_states.mean(dim=0)
        std_states = past_states.std(dim=0)
        past_reward = torch.stack(past_data.reward)

        #normalizing states
        normalized_past_states = (past_states - mean_states) / (std_states+10e-8)

        # compute std and means for RND rewards
        past_RND_rewards = (self.RNDtargetNet(normalized_past_states) - self.RNDpredictorNet(normalized_past_states))**2

        mean_RND_rewards = past_RND_rewards.mean(dim=0)
        std_RND_rewards = past_RND_rewards.std(dim=0)

        return mean_states, std_states, mean_RND_rewards, std_RND_rewards

    def training(self, number_episode, batch_size):
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
        solved_task = np.zeros((number_episode))

        #creating memory for the update of DQN
        self.replay_buffer = ReplayMemory(self.capacity, self.Transition)

        loss = np.zeros((number_episode))
        loss_per_step = ((number_episode))
        for i in tqdm(range(number_episode)):
            #variable to count env reward per ep.:
            environnement_reward_ep = 0
            #variable to count total reward per ep. :
            total_reward_ep = 0

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
                
                action = self.select_action(state, i)
                next_state, reward, terminated, truncated, _ = self.env.step(action.item())
                done = terminated or truncated
                
                environnement_reward_ep += reward
                
                # coumpte heuristic reward if chosen
                if self.use_heuristic_reward_function :
                    testing_state = state.numpy()[0]
                    
                    if action == best_next_action:
                        reward += self.heuristic_reward_function(testing_state)
                    
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
                if self.use_RND_reward and (i>27 or nb_transition_tot > 3000):

                    if mean_std_to_compute and nb_transition_tot == 3001 :#first computation of mean after 5000 transitions transitions in the first episode
                        mean_std_to_compute = False #to not enter again in if statement

                        mean_states, std_states, mean_RND_rewards, std_RND_rewards = self.init_mean_std()

                        #normalizing states
                        normalized_next_state = (next_state-mean_states)/(std_states+10e-8)
                        #compute current RND reward
                        not_normalized_RND_reward = (self.RNDtargetNet(normalized_next_state) - self.RNDpredictorNet(normalized_next_state))**2
                        RND_reward = self.normalize_RND_reward(not_normalized_RND_reward, mean_RND_rewards, std_RND_rewards)
                        
                    #normalize current state and next_state :
                    normalized_next_state = self.normalize_state(next_state, mean_states, std_states)
                    
                    if next_state is not None :

                        #updating the mean/std of past states : !!! as seen in forum discussion : no running estimate of mean/std states !!!
                        #mean_states, std_states = self.update_mean_std_states(next_state, mean_states, std_states, nb_transition_tot)

                        #compute RND reward
                        not_normalized_RND_reward = (self.RNDpredictorNet(normalized_next_state)-self.RNDtargetNet(normalized_next_state))**2

                        mean_RND_rewards, std_RND_rewards = self.update_mean_std_aux_reward(not_normalized_RND_reward, mean_RND_rewards, std_RND_rewards, nb_transition_tot)
                        RND_reward = self.normalize_RND_reward(not_normalized_RND_reward, mean_RND_rewards, std_RND_rewards)
                        clamped_RND_reward = torch.clamp(RND_reward, min=-5, max= 5).item()
                        reward = reward + clamped_RND_reward*self.global_reward_factor
                    
                total_reward_ep += reward
                
                reward = torch.tensor([reward],dtype = torch.float32)
                
                self.observe_learning(state, action, next_state, reward)

                state = next_state
                environnement_reward[i] = environnement_reward_ep
                total_reward[i] = total_reward_ep
            
            # Updating RND neural net with next_state normalized
            if (not mean_std_to_compute) and (self.use_RND_reward):
                abc = self.updateRandomNetworkDistillation(mean_states, std_states, batch_size)
            
            loss[i] = self.update(batch_size, self.lr_DQN)

            if nb_trans_ep < 200:
                tasksolve+=1 
            solved_task[i] = tasksolve
            if self.use_log : 
                self.writer.add_scalar('loss/Episode', loss[i], i)
                self.writer.add_scalar('Reward/Episode', reward.item(), i)
                self.writer.add_scalar('Nb_steps/Episode', nb_trans_ep, i)
                self.writer.add_scalar('environnement_reward/Episode', environnement_reward_ep, i)
                self.writer.add_scalar('total_reward/Episode', total_reward_ep, i)
                self.writer.add_scalar('auxiliary_reward/Episode', total_reward_ep-environnement_reward_ep, i)  
                self.writer.add_scalar('solve_task/Episode', tasksolve, i)
                self.writer.flush()
        
        auxiliary_reward = total_reward - environnement_reward
        self.nb_step_training = environnement_reward*(-1)

        return environnement_reward, auxiliary_reward, total_reward, loss, solved_task
        
    def save_nb_step_training(self):

        if self.use_RND_reward:
            np.save(f"@RND_step_training@{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}", self.nb_step_training)
        else :
            np.save(f"@heur_step_training@{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}", self.nb_step_training)
        print("saved !")
        
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
        # factor 1.1 and 0.7 to consider that the mean position is closer to the left boundary than the right booundary(goal)
        if position > -0.5:
            reward1 = abs((position-mean_position)/length_intervall_position)*1.1*1.5
            reward2 = abs((speed-mean_speed)/length_intervall_speed)*1.1
        else :
            reward1 = abs((position-mean_position)/length_intervall_position)*0.7
            reward2 = abs((speed-mean_speed)/length_intervall_speed)*0.7

        #keep max reward
        reward=min(1,max(reward1,reward2)*self.global_reward_factor)
        
        return reward
        
    def run(self, seed):
        state, _ = self.env.reset(seed = seed)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        done = False
        self.nb_step = 0 
        #do a loop if episide is not finished (truncated or terminated)
        while not done :
            #select action 
            action = self.select_action(state, iteration_number = 1, starting_epsilon = 0, ending_epsilon = 0) #select action with max(Q), no more exploration...
            action = action.item()
            #do the step relating to chosen action
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            #store obervation
            #self.observe(state, action, next_state, reward)
            # prepare next step
            state = next_state
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            done = terminated or truncated
            self.nb_step +=1
    
    def observe(self, state, action, next_state, reward):
        #store observation
        self.observed_states_speed[self.nb_step] = state[1]
        self.observed_states_position[self.nb_step] = state[0]
        self.observed_reward[self.nb_step] = reward

    def save_neural_parameter(self, use_RND=False, use_heuristic=False):
        directory = os.path.dirname(os.path.abspath(__file__))
        if use_RND:
            save_path = os.path.join(directory,"optimizeDQN_RND.pth")
        elif use_heuristic:
            save_path = os.path.join(directory,"optimizeDQN_heuristic.pth")
        else :
            save_path = os.path.join(directory,"optimizeDQN.pth")
        torch.save(self.targetNet.state_dict(),save_path)
        print(f'saved in{save_path}')