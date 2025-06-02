import numpy as np 
def _get_rew(self, x_velocity: float, action):
    # Reward for moving forward, which is the main goal
    forward_reward = self._forward_reward_weight * x_velocity
    
    # Cost incurred by action to minimize the use of energy
    action_cost = self.control_cost(action)
    
    # Cost for making contacts, if any, to avoid excessive force usage that might imply a fall or crash
    contact_penalty = self.contact_cost
    
    # Reward for maintaining a healthy state without termination
    alive_bonus = self.healthy_reward
    
    # Calculate total reward
    reward = forward_reward - action_cost - contact_penalty + alive_bonus
    
    # Prepare the reward info dictionary to debug individual components
    reward_info = {
        'forward_reward': forward_reward,
        'action_cost': action_cost,
        'contact_penalty': contact_penalty,
        'alive_bonus': alive_bonus,
        'total_reward': reward
    }
    
    return reward, reward_info
