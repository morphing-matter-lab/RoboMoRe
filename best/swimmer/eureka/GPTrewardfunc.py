import numpy as np 
def _get_rew(self, x_velocity: float, action):
    # Reward for moving right (x_velocity)
    forward_reward = self._forward_reward_weight * x_velocity

    # Cost of action use: penalize large torques to promote efficient movements
    control_cost = self.control_cost(action)

    # Calculate the total reward, encourage forward movement and discourage unnecessary movement
    reward = forward_reward - control_cost
    
    # Compile reward components into a dictionary for debugging and analysis
    reward_info = {
        'forward_reward': forward_reward,
        'control_cost': control_cost,
        'net_reward': reward
    }
    
    return reward, reward_info
