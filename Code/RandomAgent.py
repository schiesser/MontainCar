from AbstractAgent import Agent

class RandomAgent(Agent):
    
    def __init__(self, env):
        super().__init__(env)
        
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
            next_state, reward, terminated, truncated, _ = self.env.step(action)

            self.observe(state, action, next_state, reward)
            
            episode_reward += reward
            state = next_state
            done = terminated or truncated
        return episode_reward