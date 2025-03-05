import numpy as np
from src.racetrack import RaceTrack

class DynamicProgramming:
    def __init__(self, env, gamma=0.9, theta=1e-6, max_iterations=1000):
        """
        Initialize Dynamic Programming solver for RaceTrack environment.
        
        Args:
            env (RaceTrack): Instance of RaceTrack environment.
            gamma (float): Discount factor.
            theta (float): Convergence threshold.
            max_iterations (int): Maximum iterations for policy evaluation.
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations
        self.value_function = {}  # Dictionary to store state values
        self.policy = {}  # Dictionary to store policy (action per state)
        self.all_states = self._generate_all_states()
        self._initialize_policy()
        self._initialize_value_function()
    
    def _generate_all_states(self):
        """
        Generate all possible states in the environment.
        
        Returns:
            list: A list of tuples representing all valid (x, y, vx, vy) states.
        """
        states = []
        for x in range(self.env.course.shape[0]):
            for y in range(self.env.course.shape[1]):
                for vx in range(-self.env.MAX_VELOCITY, self.env.MAX_VELOCITY + 1):
                    for vy in range(-self.env.MAX_VELOCITY, self.env.MAX_VELOCITY + 1):
                        if self.env.course[x, y] != -1:  # Not a wall
                            states.append((x, y, vx, vy))
        return states

    def _initialize_policy(self):
        """
        Initialize a random policy for each state.
        """
        for state in self.all_states:
            self.policy[state] = np.random.choice(self.env.n_actions)

    def _initialize_value_function(self):
        """
        Initialize the value function to zero for all states.
        """
        for state in self.all_states:
            self.value_function[state] = 0.0
    
    def policy_evaluation(self):
        """
        Evaluate the current policy by iteratively updating the value function.
        
        This function should update self.value_function in place until convergence.
        
        Returns:
            None
        """
        # TODO: Implement policy evaluation using iterative updates

        for _ in range(self.max_iterations):
            delta = 0  # Track changes in value function
            new_value_function = self.value_function.copy()  # Copy of the value function for updates

            for state in self.all_states:
                action = self.policy[state]  # Get the action from the current policy
                value = 0  # Initialize value update

                for prob, next_state, reward, done in self.env.get_transitions(state, action):
                    value += prob * (reward + self.gamma * self.value_function[next_state] * (not done))
                
                delta = max(delta, abs(value - self.value_function[state]))  # Track the max change
                new_value_function[state] = value  # Update value function

            self.value_function = new_value_function  # Update value function after sweep

            if delta < self.theta:  # Stop if value function has converged
                break

    
    def policy_improvement(self):
        """
        Improve the policy using the current value function.
        
        Returns:
            bool: True if the policy is stable (no changes), False otherwise.
        """
        # TODO: Implement policy improvement by choosing the best action
    
        policy_stable = True  # Track if policy changes

        for state in self.all_states:
            old_action = self.policy[state]  # Store the current action
            
            # Find the best action by maximizing the expected value
            action_values = {}
            for action in range(self.env.n_actions):
                action_value = 0
                for prob, next_state, reward, done in self.env.get_transitions(state, action):
                    action_value += prob * (reward + self.gamma * self.value_function[next_state] * (not done))
                action_values[action] = action_value

            best_action = max(action_values, key=action_values.get)  # Choose the action with the highest value

            # Update policy if the best action is different from the old one
            if old_action != best_action:
                policy_stable = False
                self.policy[state] = best_action

        return policy_stable  # Return whether the policy remained unchanged

    
    def policy_iteration(self):
        """
        Perform Policy Iteration algorithm.
        
        This function should alternate between policy evaluation and policy improvement
        until convergence.
        
        Returns:
            None
        """
        # TODO: Implement Policy Iteration
        while True:
            self.policy_evaluation()  # Step 1: Evaluate the current policy
            policy_stable = self.policy_improvement()  # Step 2: Improve the policy

            if policy_stable:
                break  # Stop if policy remains unchanged
    
    def value_iteration(self):
        """
        Perform Value Iteration algorithm.
        
        This function should update the value function and extract the optimal policy.
        
        Returns:
            None
        """
        # TODO: Implement Value Iteration
        for _ in range(self.max_iterations):
            delta = 0  # Track the maximum change in value function
            new_value_function = self.value_function.copy()

            for state in self.all_states:
                action_values = []
                for action in range(self.env.n_actions):
                    action_value = 0
                    for prob, next_state, reward, done in self.env.get_transitions(state, action):
                        action_value += prob * (reward + self.gamma * self.value_function[next_state] * (not done))
                    action_values.append(action_value)

                best_value = max(action_values)  # Find the best action value
                delta = max(delta, abs(best_value - self.value_function[state]))  # Track the max update
                new_value_function[state] = best_value  # Update value function

            self.value_function = new_value_function  # Apply updates to the value function

            if delta < self.theta:  # Stop if the value function has converged
                break

        # Extract the optimal policy after convergence
        for state in self.all_states:
            action_values = {}
            for action in range(self.env.n_actions):
                action_value = 0
                for prob, next_state, reward, done in self.env.get_transitions(state, action):
                    action_value += prob * (reward + self.gamma * self.value_function[next_state] * (not done))
                action_values[action] = action_value
            
            self.policy[state] = max(action_values, key=action_values.get)  # Select best action

    
    def _simulate_action(self, state, action):
        """
        Simulate taking an action from a given state.
        
        Args:
            state (tuple): The current state (x, y, vx, vy).
            action (int): The action to take.
        
        Returns:
            tuple: (reward, new_state) where new_state is the next state tuple.
        """
        x, y, vx, vy = state
        self.env.position = np.array([x, y])
        self.env.velocity = np.array([vx, vy])
        reward = self.env.take_action(int(action))
        new_state = tuple(self.env.get_state())
        return reward, new_state
    
    def solve(self, method='policy_iteration'):
        """
        Solve the environment using the specified DP method.
        
        Args:
            method (str): 'policy_iteration' or 'value_iteration'.
        
        Returns:
            None
        """
        if method == 'policy_iteration':
            self.policy_iteration()
        elif method == 'value_iteration':
            self.value_iteration()
        else:
            raise ValueError("Invalid method. Choose 'policy_iteration' or 'value_iteration'")
    
    def print_policy(self):
        """
        Print the policy in a readable format.
        """
        for y in range(self.env.course.shape[1] - 1, -1, -1):
            row = ''
            for x in range(self.env.course.shape[0]):
                state = (x, y, 0, 0)  # Default velocity
                if state in self.policy:
                    row += str(self.policy[state]) + ' '
                else:
                    row += 'W ' if self.env.course[x, y] == -1 else '. '
            print(row)

# Test the implementation
if __name__ == "__main__":
    tiny_course = [
        "WWWWWW",
        "Woooo+",
        "Woooo+",
        "WooWWW",
        "WooWWW",
        "WooWWW",
        "WooWWW",
        "W--WWW",
    ]
    env = RaceTrack(tiny_course)  # Use the tiny race track for testing
    dp_solver = DynamicProgramming(env)
    dp_solver.solve(method='policy_iteration')  # Change to 'value_iteration' if needed
    dp_solver.print_policy()
