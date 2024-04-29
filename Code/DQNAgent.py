from AbstractAgent import Agent

class DQNAgent(Agent):
    
    def __init__(self, batch_size = 64, discount_factor = 0.99):
        super().__init__()
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        
    def select_action(self, env, epsilon, Q):
        # e-greedy, that can vary (see doc)
        """
        Chooses an epsilon-greedy action starting from a given state and given Q-values
        :param env: environment
        :param epsilon: current exploration parameter
        :param Q: current Q-values.
        :return:
            - the chosen action
        """
        raise NotImplementedError
        
        # get the available actions
        available_actions = self.action_space
    
        if np.random.uniform(0, 1) < epsilon:
            # with probability epsilon make a random move (exploration)
            return np.random.choice(available_actions)
        else:
            # with probability 1-epsilon choose the action with the highest immediate reward (exploitation)
            q = np.copy(Q[env.get_state()]) # for us state=position and speed : continuous
            mask = [env.encode_action(action) for action in available_actions]
            q = [q[i] if i in mask else np.nan for i in range(len(q))]
            max_indices = np.argwhere(q == np.nanmax(q)).flatten()  # best action(s) along the available ones
            return env.inverse_encoding(int(np.random.choice(max_indices)))  # ties are split randomly


    def Q(self, state, action):
        # MLP
        raise NotImplementedError
    
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