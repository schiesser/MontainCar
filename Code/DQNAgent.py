from AbstractAgent import Agent
from DQN import DQN
import math

class DQNAgent(Agent):
    
    def __init__(self, discount_factor = 0.99):
        super().__init__()
        self.discount_factor = discount_factor
        
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
    
    def observe(self, state, action, next_state, reward): #replay buffer
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
        return episode_reward