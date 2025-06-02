import numpy as np 
def _get_rew(self, x_velocity: float, action):
    # Reward component based on the velocity of the cheetah moving forward
    forward_reward = self._forward_reward_weight * x_velocity 
    
    # Control cost to discourage excessive usage of force
    control_cost = self.control_cost(action)
    
    # Total reward is a balance between motivating high speeds and penalizing excessive control efforts
    reward = forward_reward - control_cost
    
    # Reward info dictionary for debugging and monitoring individual components
    reward_info = {
        'forward_reward': forward_reward,
        'control_cost': control_cost
    }
    
    return reward, reward_info
