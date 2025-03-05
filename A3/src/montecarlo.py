import numpy as np
import random
from collections import defaultdict
from src.racetrack import RaceTrack

class MonteCarloControl:
    """
    Monte Carlo Control with Weighted Importance Sampling for off-policy learning.
    
    This class implements the off-policy every-visit Monte Carlo Control algorithm
    using weighted importance sampling to estimate the optimal policy for a given
    environment.
    """
    def __init__(self, env: RaceTrack, gamma: float = 1.0, epsilon: float = 0.1, Q0: float = 0.0, max_episode_size : int = 1000):
        """
        Initialize the Monte Carlo Control object. 

        Q, C, and policies are defaultdicts that have keys representing environment states.  
        Defaultdicts (search up the docs!) allow you to set a sensible default value 
        for the case of Q[new state never visited before] (and likewise with C/policies).  
        

        Hints: 
        - Q/C/*_policy should be defaultdicts where the key is the state
        - each value in the dict is a numpy vector where position is indexed by action
        - That is, these variables are setup like Q[state][action]
        - state key will be the numpy state vector cast to string (dicts require hashable keys)
        - Q should default to Q0, C should default to 0
        - *_policy should default to equiprobable (random uniform) actions
        - store everything as a class attribute:
            - self.env, self.gamma, self.Q, etc...

        Args:
            env (racetrack): The environment in which the agent operates.
            gamma (float): The discount factor.
            Q0 (float): the initial Q values for all states (e.g. optimistic initialization)
            max_episode_size (int): cutoff to prevent running forever during MC
        
        Returns: none, stores data as class attributes
        """
         # Your code here
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_episode_size = max_episode_size

        # Q: Q-values initialized to Q0 (state-action dictionary)
        self.Q = defaultdict(lambda: np.full(self.env.n_actions, Q0))

        # C: Cumulative sum of importance sampling weights
        self.C = defaultdict(lambda: np.zeros(self.env.n_actions))

        # Target policy (greedy policy)
        self.target_policy = defaultdict(lambda: np.zeros(self.env.n_actions))

        # Behavior policy (epsilon-greedy policy)
        self.behavior_policy = defaultdict(lambda: np.ones(self.env.n_actions) / self.env.n_actions)

        # Initialize the policies to be equiprobable at all states
        for state in self.env.get_all_states():
            self.target_policy[state] = np.ones(self.env.n_actions) / self.env.n_actions
            self.behavior_policy[state] = np.ones(self.env.n_actions) / self.env.n_actions



    def create_target_greedy_policy(self):
        """
        Loop through all states in the self.Q dictionary. 
        1. determine the greedy policy for that state
        2. create a probability vector that is all 0s except for the greedy action where it is 1
        3. store that probability vector in self.target_policy[state]

        Args: none
        Returns: none, stores new policy in self.target_policy
        """
        # Your code here
        for state, action_values in self.Q.items():
            # Find the action with the highest Q-value (greedy action)
            best_action = np.argmax(action_values)

            # Create a probability vector with all zeros except the best action set to 1
            greedy_policy = np.zeros(self.env.n_actions)
            greedy_policy[best_action] = 1.0

            # Store the new policy for this state
            self.target_policy[state] = greedy_policy


    def create_behavior_egreedy_policy(self):
        """
        Loop through all states in the self.target_policy dictionary. 
        Using that greedy probability vector, and self.epsilon, 
        calculate the epsilon greedy behavior probability vector and store it in self.behavior_policy[state]
        
        Args: none
        Returns: none, stores new policy in self.target_policy
        """
        # Your code here
        n_actions = self.env.n_actions  # Number of available actions

        for state, greedy_policy in self.target_policy.items():
            # Get the index of the greedy action
            best_action = np.argmax(greedy_policy)

            # Initialize epsilon-greedy policy
            egreedy_policy = np.ones(n_actions) * (self.epsilon / n_actions)  # Small probability for all actions

            # Assign the remaining probability to the greedy action
            egreedy_policy[best_action] += (1 - self.epsilon)

            # Store the epsilon-greedy policy
            self.behavior_policy[state] = egreedy_policy



        
    def egreedy_selection(self, state):
        """
        Select an action proportional to the probabilities of epsilon-greedy encoded in self.behavior_policy
        HINT: 
        - check out https://www.w3schools.com/python/ref_random_choices.asp
        - note that random_choices returns a numpy array, you want a single int
        - make sure you are using the probabilities encoded in self.behavior_policy 

        Args: state (string): the current state in which to choose an action
        Returns: action (int): an action index between 0 and self.env.n_actions
        """
        # Your code here
        action_probs = self.behavior_policy[state]

        # Select an action based on the probability distribution
        action = random.choices(range(self.env.n_actions), weights=action_probs)[0]

        return action


    def generate_egreedy_episode(self):
        """
        Generate an episode using the epsilon-greedy behavior policy. Will not go longer than self.max_episode_size
        
        Hints: 
        - need to setup and use self.env methods and attributes
        - use self.egreedy_selection() above as a helper function
        - use the behavior e-greedy policy attribute aleady calculated (do not update policy here!)
        
        Returns:
            list: The generated episode, which is a list of (state, action, reward) tuples.
        """
        # Your code here

        episode = []  # Stores (state, action, reward) tuples
        state = self.env.reset()  # Reset the environment and get initial state

        for _ in range(self.max_episode_size):
            state_str = str(state)  # Convert state to string for dictionary indexing
            action = self.egreedy_selection(state_str)  # Select action using epsilon-greedy policy
            next_state, reward, done = self.env.step(action)  # Take action and observe next state, reward, done
            
            episode.append((state_str, action, reward))  # Store experience

            if done:  # End episode if terminal state is reached
                break

            state = next_state  # Move to the next state

        return episode

    
    def generate_greedy_episode(self):
        """
        Generate an episode using the greedy target policy. Will not go longer than self.max_episode_size
        Note: this function is not used during learning, its only for evaluating the target policy
        
        Hints: 
        - need to setup and use self.env methods and attributes
        - use the greedy policy attribute aleady calculated (do not update policy here!)

        Returns:
            list: The generated episode, which is a list of (state, action, reward) tuples.
        """
        # Your code here

        episode = []  # Stores (state, action, reward) tuples
        state = self.env.reset()  # Reset the environment and get initial state

        for _ in range(self.max_episode_size):
            state_str = str(state)  # Convert state to string for dictionary indexing
            action = np.argmax(self.target_policy[state_str])  # Select the greedy action
            next_state, reward, done = self.env.step(action)  # Take action and observe next state, reward, done
            
            episode.append((state_str, action, reward))  # Store experience

            if done:  # End episode if terminal state is reached
                break

            state = next_state  # Move to the next state

        return episode

    
    def update_offpolicy(self, episode):
        """
        Update the Q-values using every visit weighted importance sampling. 
        See Figure 5.9, p. 134 of Sutton and Barto 2nd ed.
        
        Args: episode (list): An episode generated by the behavior policy; a list of (state, action, reward) tuples.
        Returns: none
        """
        # Your code here

        G = 0  # Initialize return (cumulative reward)
        W = 1  # Importance sampling weight (starts at 1)

        # Iterate backward through the episode
        for state, action, reward in reversed(episode):
            G = self.gamma * G + reward  # Compute return

            # Update cumulative sum of importance sampling weights
            self.C[state][action] += W

            # Update Q-value estimate using weighted importance sampling
            self.Q[state][action] += (W / self.C[state][action]) * (G - self.Q[state][action])

            # Stop updating if the behavior policy deviates from the target policy
            if self.behavior_policy[state][action] == 0:
                break  # Avoid division by zero, meaning the action was never selected

            # Update importance sampling weight
            W *= self.target_policy[state][action] / self.behavior_policy[state][action]


    
    def update_onpolicy(self, episode):
        """
        Update the Q-values using first visit epsilon-greedy. 
        See Figure 5.6, p. 127 of Sutton and Barto 2nd ed.
        
        Args: episode (list): An episode generated by the behavior policy; a list of (state, action, reward) tuples.
        Returns: none
        """
        # Your code here
        G = 0  # Initialize cumulative return
        visited = set()  # Track first visits

        # Iterate backward through the episode
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            state_action_pair = (state, action)

            G = self.gamma * G + reward  # Compute return

            # First-visit check: Update only if (state, action) has not been visited before
            if state_action_pair not in visited:
                visited.add(state_action_pair)  # Mark as visited
                self.C[state][action] += 1  # Increment counter
                self.Q[state][action] += (1 / self.C[state][action]) * (G - self.Q[state][action])  # Incremental update

                # Update policy to be epsilon-greedy
                best_action = np.argmax(self.Q[state])  # Get greedy action
                self.behavior_policy[state] = np.ones(self.env.n_actions) * (self.epsilon / self.env.n_actions)  # Uniform probability
                self.behavior_policy[state][best_action] += (1 - self.epsilon)  # Assign most weight to the best action



    def train_offpolicy(self, num_episodes):
        """
        Train the agent over a specified number of episodes.
        
        Args:
            num_episodes (int): The number of episodes to train the agent.
        """
        for _ in range(num_episodes):
            episode = self.generate_egreedy_episode()
            self.update_offpolicy(episode)

   


    def get_greedy_policy(self):
        """
        Retrieve the learned target policy in the form of an action index per state
        
        Returns:
            dict: The learned target policy.
        """
        policy = {}
        for state, actions in self.Q.items():
            policy[state] = np.argmax(actions)
        return policy