import numpy as np
def _get_rew(self, x_velocity: float, action):
    # Reward for moving towards the right along the x-axis
    forward_reward = self._forward_reward_weight * x_velocity

    # Penalty for using the rotors, i.e., minimize control effort to avoid too much spinning and erratic behavior
    control_penalty = self.control_cost(action)

    # Total reward is the forward reward minus the control costs
    total_reward = forward_reward - control_penalty
    
    # Additional reward info, helping in debugging or further analysis
    reward_info = {
        'forward_reward': forward_reward,
        'control_penalty': control_penalty,
    }
    
    return total_reward, reward_info
