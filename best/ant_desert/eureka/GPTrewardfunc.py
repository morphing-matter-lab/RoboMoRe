import numpy as np 
def _get_rew(self, x_velocity: float, action):
    # Calculate the reward for moving forward.
    # The agent gets a higher reward for higher x_velocity
    forward_reward = self._forward_reward_weight * x_velocity
    
    # Penalize the agent for using too much control power to execute actions. 
    # This encourages the agent to move efficiently.
    control_cost = self.control_cost(action)

    # Calculate contact cost to penalize the agent for excessive contact with the ground, 
    # which might indicate dragging or inefficient locomotion.
    contact_cost = self.contact_cost

    # Compute total reward by adding forward_reward, the healthy_reward, and subtracting costs.
    total_reward = forward_reward + self.healthy_reward - control_cost - contact_cost
    
    # Construct reward info dictionary
    reward_info = {
        'forward_reward': forward_reward,
        'healthy_reward': self.healthy_reward,
        'control_cost': control_cost,
        'contact_cost': contact_cost
    }
    
    return total_reward, reward_info
