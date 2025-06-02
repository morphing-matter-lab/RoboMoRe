import numpy as np 
def _get_rew(self, x_velocity: float, action):
    # Encourage the walker to move as fast as possible in the x direction
    speed_reward = self._forward_reward_weight * x_velocity

    # Penalize the use of excessive control inputs to prevent erratic behavior
    control_penalty = self.control_cost(action)

    # A healthy bonus ensures agent stays upright and within angle and z-pos limits
    health_bonus = self.healthy_reward

    # Total reward
    reward = speed_reward - control_penalty + health_bonus
    
    # Information dictionary for individual components of the reward
    reward_info = {
        'speed_reward': speed_reward,
        'control_penalty': control_penalty,
        'health_bonus': health_bonus
    }

    return reward, reward_info
