import numpy as np 
def _get_rew(self, x_velocity: float, action):
    # Defining the main components of the reward function
    forward_reward = self._forward_reward_weight * x_velocity  # Encourages moving forward
    control_cost = self.control_cost(action)  # Penalizes the magnitude of action to encourage efficient movements
    contact_cost = self.contact_cost  # Penalizes the magnitude of contact forces to encourage smooth locomotion
    
    # Combining the reward components
    reward = forward_reward - control_cost - contact_cost + self.healthy_reward  # Adding healthy_reward encourages staying upright
    
    # Information about the reward components for monitoring and debugging
    reward_info = {
        'forward_reward': forward_reward,
        'control_cost': control_cost,
        'contact_cost': contact_cost,
        'healthy_reward': self.healthy_reward
    }
    
    return reward, reward_info
