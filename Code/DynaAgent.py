from AbstractAgent import Agent

import numpy as np
import math
import random
import torch
import time
import datetime
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from replay_buffer import ReplayMemoryDyna
from collections import namedtuple

class DynaAgent(Agent):
    
    def __init__(self, env, discr_step = [0.025, 0.005], discount_factor = 0.99, k_updates = 10, should_log = True):
        super().__init__(env)

        # Environment values
        self.nb_actions = 3
        self.interval_position = [-1.2, 0.6]
        self.interval_speed = [-0.07, 0.07]

        # Initialization variables
        self.k_updates = k_updates
        self.discr_step = discr_step
        self.discount_factor = discount_factor
        self.should_log = should_log

        # Calculate the number of states per interval/velocity and number of states
        self.nb_interval_position = int(abs(self.interval_position[0] - self.interval_position[1]) / discr_step[0] )
        self.nb_interval_speed = int(abs(self.interval_speed[0] - self.interval_speed[1]) / discr_step[1] )
        self.nb_states = self.nb_interval_position * self.nb_interval_speed

        # Discretization of position and speed
        self.discretization_position = np.linspace(self.interval_position[0], self.interval_position[1], self.nb_interval_position + 1)
        self.discretization_speed = np.linspace(self.interval_speed[0], self.interval_speed[1], self.nb_interval_speed + 1)

        # Tabular values :
        self.P_estimate = np.ones((self.nb_states, self.nb_actions, self.nb_states)) / (self.nb_states * self.nb_actions)
        self.R_estimate = np.zeros((self.nb_states, self.nb_actions))
        self.Q = np.zeros((self.nb_states, self.nb_actions))
        
        self.Cnt = np.zeros((self.nb_states, self.nb_actions, self.nb_states))
        self.cumv_R = np.zeros((self.nb_states, self.nb_actions))

        print(f"-----------variables- new--------")
        print(f"number of positions = {self.nb_interval_position}")
        print(f"number of velocity = {self.nb_interval_speed}")
        print(f"number of states = {self.nb_states}")
        print(f"self.discretization_position = {self.discretization_position}")
        print(f"self.discretization_speed = {self.discretization_speed}")
        print(f"-----------------------------------------")

        self.freq_result = [0, 0, 0]

        # Writer for logging purpose
        if self.should_log:
            logdir = f'./runs/new@discr_step={str(discr_step)}@discount_factor={discount_factor}@k_updates={k_updates}@{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
            self.writer = SummaryWriter(log_dir=logdir)
            print(f"------------------------------------------\nWe will log this experiment in directory {logdir}\n------------------------------------------")

        # Replay buffer
        print("new replaybuffer")
        self.Transition = namedtuple('Transition',('state', 'action'))
        self.replay_buffer = ReplayMemoryDyna(k_updates, self.Transition)

        self.freq_actions = [0, 0, 0]
        
        
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
        indice_position = int((correct_indice - indice_speed) / self.nb_interval_speed)

        position = np.array([self.discretization_position[indice_position], self.discretization_position[indice_position] + 1])
        speed = np.array([self.discretization_speed[indice_speed], self.discretization_speed[indice_speed] + 1])

        return position,speed


    def select_action(self, state, iteration_number, starting_epsilon = 0.9, ending_epsilon = 0.05, epsilon_decay = 150):
        """
        Chooses an epsilon-greedy action given a state and its Q-values associated
        parameters : starting_epsilon, ending_epsilon, epsilon_decay allow to manage exploitation and exploration during action selection
        epsilon is decresing relatively to the number of iterations
        """
        # reason : more exploration at beginning, and increase exploitation with iteration
        epsilon = ending_epsilon + (starting_epsilon - ending_epsilon) * math.exp(-iteration_number/epsilon_decay)
        indice_state = self.found_indice_bin(state) 
        
        if np.random.uniform(0, 1) > epsilon:
            result = np.argmax(self.Q[indice_state,:])
            self.freq_result[result] += 1
            return result
        else :
            return np.random.choice(np.array([0, 1, 2]))
    
        
    def observe(self, state, action, next_state, reward):
        index_s = self.found_indice_bin(state)
        index_s_prime = self.found_indice_bin(next_state)

        # update the counters
        self.Cnt[index_s][action][index_s_prime] += 1
        self.cumv_R[index_s][action] += reward

        sum_cnt_s_a = np.sum(self.Cnt[index_s][action])

        #update the reward and probabilities
        self.R_estimate[index_s][action] = self.cumv_R[index_s][action] / sum_cnt_s_a
        
        # for index_next_state in range(self.nb_states):
        #     self.P_estimate[index_s][action][index_next_state] = self.Cnt[index_s][action][index_next_state] / sum_cnt_s_a
        self.P_estimate[index_s][action] = self.Cnt[index_s][action] / sum_cnt_s_a

        expected_future_reward = np.sum(np.multiply(self.P_estimate[index_s][action], np.max(self.Q, axis=-1)))
        self.Q[index_s][action] = self.R_estimate[index_s][action] + self.discount_factor * expected_future_reward
        

    def update(self, iteration_number):
        if iteration_number > self.k_updates - 1:
            list_transitions = self.replay_buffer.sample()

            for transition in list_transitions:
                s = self.found_indice_bin(transition.state)
                a = transition.action

                expected_future_reward = np.sum(np.multiply(self.P_estimate[s][a], np.max(self.Q, axis=-1)))
                self.Q[s][a] = self.R_estimate[s][a] + expected_future_reward * self.discount_factor
        


    def run(self, num_episodes = 3000, starting_epsilon = 0.9, ending_epsilon = 0.05, epsilon_decay = 150):
        total_reward = []
        tasksolve = 0
        for episode in tqdm(range(num_episodes)):
            start_time = time.time()
            update_time_total = 0
            rew_ep = 0
            num_steps = 0
            self.observations = []
            state, _ = self.env.reset()
            done = False
            while not done:
                num_steps = num_steps + 1
                action = self.select_action(state=state, 
                                            iteration_number=episode, 
                                            starting_epsilon=starting_epsilon,
                                            ending_epsilon=ending_epsilon, 
                                            epsilon_decay=epsilon_decay
                                            )
                self.freq_actions[action] += 1
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                self.observe(state, action, next_state, reward)
                if terminated:
                    print(f"Episode {episode} is done successfully !!!!!!")
                done = terminated or truncated
                if reward != -1.0:
                    print(f"reward = {reward} type = {type(reward)}")
                rew_ep = rew_ep + reward
                self.observations.append(next_state)
                update_time_start = time.time()
                self.update(iteration_number=episode)
                update_time_total += (time.time() - update_time_start)
                self.replay_buffer.push(state, action)
                state = next_state
            if num_steps < 200:
                tasksolve += 1
            if self.should_log:
                self.writer.add_scalar('Reward/Episode', rew_ep, episode)
                self.writer.add_scalar('Nb_steps/Episode', num_steps, episode)
                self.writer.add_scalar('Solve_Task/Episode', tasksolve, episode)
                self.writer.add_scalar('Seconds/Episode', (time.time() - start_time), episode)
                self.writer.add_scalar('Seconds_Update/Episode', update_time_total, episode)
                self.writer.flush()
            total_reward.append(rew_ep)
        return total_reward