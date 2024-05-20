from AbstractAgent import Agent

class DynaAgent(Agent):
    
    def __init__(self, env, nb_interval, discount_factor = 0.99, k = 1):
        
        super().__init__(env)
        self.nb_interval_position = nb_interval[0]
        self.nb_interval_speed = nb_interval[1]
        nb_states = self.nb_interval_position*self.nb_interval_speed
        nb_actions = 3
        
        self.discount_factor = discount_factor

        interval_position = [-1.2, 0.6]
        interval_speed = [-0.07, 0.007]
        

        #think better to have nb_interval instead of a step : if use step -> use arange...
        self.discretization_position = np.linspace(interval_position[0], interval_position[1], self.nb_interval_position[0]+1)
        self.discretization_speed = np.linspace(interval_v[0], interval_speed[1], self.nb_interval_speed[1]+1)

        #attention indexing !!! discuss tomorrow
        self.P_estimate = np.ones((nb_states, nb_actions, nb_states))*(1/nb_states) #uniform at initialization
        self.R_estimate = np.zeros((nb_states, nb_actions))
        self.Q = np.zeros((nb_states, nb_actions))
        
    def found_indice_bin(state):
    
        position = state[0]
        speed = state[1]

        indice_position = np.digitize(position, self.discretization_position)-1 #indices between 0 and (nb_interval-1), directly good to slice numpy vector !
        indice_speed = np.digitize(speed, self.discretization_speed)-1

        correct_indice = indice_position*self.nb_interval_speed + indice_speed
        
        return correct_indice
        
    def found_state(correct_indice): #not sure it will be usefull
        
        indice_speed = correct_indice % self.nb_interval_speed
        indice_position = int((correct_indice - indice_speed)/self.nb_interval_speed)

        position = np.array([self.discretization_position[indice_position], self.discretization_position[indice_position]+1])
        speed = np.array([self.discretization_speed[indice_speed], self.discretization_speed[indice_speed]+1])

        return position,speed

    def select_action(self, state, iteration_number, starting_epsilon = 0.8, ending_epsilon = 0.05, epsilon_decay = 150):
        """
        Chooses an epsilon-greedy action given a state and its Q-values associated
        parameters : starting_epsilon, ending_epsilon, epsilon_decay allow to manage exploitation and exploration during action selection
        epsilon is decresing relatively to the number of iterations
        """
        # reason : more exploration at beginning, and increase exploitation with iteration
        epsilon = ending_epsilon + (starting_epsilon - ending_epsilon) * math.exp(-iteration_number/epsilon_decay)

        indice_state = found_indice_bin(state) 
        
        if np.random.uniform(0, 1) > epsilon:
            best_action = np.max(self.Q[indice_state,:])
            return best_action
        else :
            random_action = np.random.choice(self.Q[indice_state,:])# choose a random action
            return random_action
    
    def run():
                