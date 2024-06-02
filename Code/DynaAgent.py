import numpy as np
import math
import time
import os
import json
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import namedtuple
from torch.utils.tensorboard import SummaryWriter
from replay_buffer import ReplayMemoryDyna
from AbstractAgent import Agent

class DynaAgent(Agent):
    
    def __init__(self, env, discr_step = [0.025, 0.005], discount_factor = 0.99, k_updates = 10, should_log = True, load_model = False):
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
        self.nb_interval_position = int(1.8 / discr_step[0] )
        self.nb_interval_speed = int(0.14 / discr_step[1] )
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

        # print(f"-----------variables- new--------")
        # print(f"number of positions = {self.nb_interval_position}")
        # print(f"number of velocity = {self.nb_interval_speed}")
        # print(f"number of states = {self.nb_states}")
        # print(f"self.discretization_position = {self.discretization_position}")
        # print(f"self.discretization_speed = {self.discretization_speed}")
        # print(f"-----------------------------------------")

        # Writer for logging purpose
        if self.should_log:
            logdir = f'./runs/new@discr_step={str(discr_step)}@discount_factor={discount_factor}@k_updates={k_updates}@{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
            self.writer = SummaryWriter(log_dir=logdir)
            print(f"------------------------------------------\nWe will log this experiment in directory {logdir}\n------------------------------------------")

        # Replay buffer
        self.Transition = namedtuple('Transition',('state', 'action'))
        self.replay_buffer = ReplayMemoryDyna(k_updates, self.Transition)

        if load_model:
            self.load_model()

        # Store some metrics for visualization/plots
        self.freq_result = [0, 0, 0]
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
        

    def found_state(self, correct_indice):

        indice_speed = correct_indice % self.nb_interval_speed
        indice_position = int((correct_indice - indice_speed) / self.nb_interval_speed)

        position = np.array([self.discretization_position[indice_position], self.discretization_position[indice_position] + 1])
        speed = np.array([self.discretization_speed[indice_speed], self.discretization_speed[indice_speed] + 1])

        return position,speed


    def select_action(self, state, iteration_number, starting_epsilon = 0.9, ending_epsilon = 0.05, epsilon_decay = 150):
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
        self.P_estimate[index_s][action] = self.Cnt[index_s][action] / sum_cnt_s_a

        expected_future_reward = np.sum(np.multiply(self.P_estimate[index_s][action], np.max(self.Q, axis=-1)))
        old_q = self.Q[index_s][action]
        self.Q[index_s][action] = self.R_estimate[index_s][action] + self.discount_factor * expected_future_reward
        # we want delta Q
        return abs(self.Q[index_s][action] - old_q)
        

    def update(self, iteration_number):
        if iteration_number > self.k_updates - 1:
            list_transitions = self.replay_buffer.sample()

            for transition in list_transitions:
                s = self.found_indice_bin(transition.state)
                a = transition.action

                expected_future_reward = np.sum(np.multiply(self.P_estimate[s][a], np.max(self.Q, axis=-1)))
                self.Q[s][a] = self.R_estimate[s][a] + expected_future_reward * self.discount_factor
        


    def train(self, num_episodes = 3000, starting_epsilon = 0.9, ending_epsilon = 0.05, epsilon_decay = 150):
        self.delta_q_update = []
        total_reward = []
        tasksolve = 0

        #create a picture directory to store the plot of q values every 100 ep
        picture_dir = f'./dyna_pics/discr_step={str(self.discr_step)}@discount_factor={self.discount_factor}@{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
        os.mkdir(picture_dir)

        for episode in tqdm(range(num_episodes)):
            if episode % 100 == 0:
                self.plot_q_value(episode=episode, picture_dir=picture_dir)

            # Initialize the episode variables
            self.observations, start_time, reward_episode, num_steps, delta_q = [], time.time(), 0, 0, 0
            
            # Reset the environment
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
                delta_q += self.observe(state, action, next_state, reward)

                done = terminated or truncated
                self.observations.append(next_state)

                self.update(iteration_number=episode)

                reward_episode = reward_episode + reward

                self.replay_buffer.push(state, action)
                state = next_state
            # While ends here => episode is finished
            
            total_reward.append(reward_episode)
            self.delta_q_update.append(delta_q)
            if num_steps < 200:
                tasksolve += 1

            #Log realtime the metrics
            if self.should_log:
                self.writer.add_scalar('Delta_Q/Episode', delta_q, episode)
                self.writer.add_scalar('Reward/Episode', reward_episode, episode)
                self.writer.add_scalar('Nb_steps/Episode', num_steps, episode)
                self.writer.add_scalar('Solve_Task/Episode', tasksolve, episode)
                self.writer.add_scalar('Seconds/Episode', (time.time() - start_time), episode)
                self.writer.flush()
        self.save_model()
        return total_reward
    
    def run(self, seed):
        state, _ = self.env.reset(seed = seed)
        done = False
        self.test_nb_step = 0
        self.test_reward_episode = 0
        
        while not done :
            action = self.select_action(state, iteration_number=1, starting_epsilon=0, ending_epsilon=0)
            
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            
            done = terminated or truncated

            self.test_nb_step += 1
            self.test_reward_episode += reward

            state = next_state
        
        if self.test_nb_step < 200:
            return True
    

    def save_model(self, dir = "./dyna_models"):
        model_name = f'discr_step={str(self.discr_step)}@discount_factor={self.discount_factor}@{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
        logdir = f"{dir}/{model_name}"
        os.mkdir(logdir)
        os.chdir(logdir)

        # Save the numpy arrays
        np.save("Cnt", self.Cnt)
        np.save("cumv_R", self.cumv_R)
        np.save("Q", self.Q)
        np.save("R_estimate", self.R_estimate)
        np.save("P_estimate", self.P_estimate)

        # Save other parameters
        params = {}
        params["discount_factor"] = self.discount_factor
        params["discr_step"] = self.discr_step

        with open("params.json", 'w') as f:
            json.dump(params, f)

        print(f"[Save Model] : Model was saved succesfully in {model_name} !")
        

    def load_model(self, dir = "./dyna_models", model_name = "discr_step=[0.1, 0.01]@discount_factor=0.99@20240601-110835"):
        os.chdir(dir + "/" + model_name)

        self.Cnt = np.load("Cnt.npy")
        self.cumv_R = np.load("cumv_R.npy")
        self.Q = np.load("Q.npy")
        self.R_estimate = np.load("R_estimate.npy")
        self.P_estimate = np.load("P_estimate.npy")

        with open("params.json", 'r') as f:
            params = json.load(f)
            self.discount_factor = params["discount_factor"]
            self.discr_step = params["discr_step"]

        # Re-do the intervals accordingly to discr_step from json
        
        # Calculate the number of states per interval/velocity and number of states
        self.nb_interval_position = int(1.8 / self.discr_step[0] )
        self.nb_interval_speed = int(0.14 / self.discr_step[1] )
        self.nb_states = self.nb_interval_position * self.nb_interval_speed

        # Discretization of position and speed
        self.discretization_position = np.linspace(self.interval_position[0], self.interval_position[1], self.nb_interval_position + 1)
        self.discretization_speed = np.linspace(self.interval_speed[0], self.interval_speed[1], self.nb_interval_speed + 1)

        print(f"[Load Model] : Model was loaded succesfully from {model_name} !")

    def plot_q_value(self, episode, picture_dir):
        # do the plotting 
        position_bins = np.arange(-1.2, 0.6, self.discr_step[0])
        velocity_bins = np.arange(-0.07, 0.0699, self.discr_step[1])
        
        num_pos_bins = len(position_bins)
        num_vel_bins = len(velocity_bins)

        max_Q_values = np.max(self.Q, axis=1).reshape(num_pos_bins, num_vel_bins)
        unvisited_mask = np.all(self.Q == 0, axis=1).reshape(num_pos_bins, num_vel_bins)
        masked_max_Q_values = np.ma.masked_where(unvisited_mask, max_Q_values)

        # Plot the max Q-values as a heatmap
        plt.figure(figsize=(12, 8))
        heatmap = plt.imshow(masked_max_Q_values.T, cmap='plasma', origin='lower', aspect='auto',
                     extent=[position_bins[0], position_bins[-1] + self.discr_step[0], velocity_bins[0], velocity_bins[-1] + self.discr_step[1]])
        plt.colorbar(heatmap, label='Max Q-value')
        plt.xlabel('Position')
        plt.ylabel('Velocity')
        plt.title(f'Max Q-value for each (Position, Velocity) discr_step={self.discr_step} episode={episode}')
        plt.savefig(f'{picture_dir}/Dyna_Q_values@discr_step={self.discr_step}@episode={episode}.png') 
        plt.close()
