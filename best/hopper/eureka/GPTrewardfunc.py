import numpy as np 
def _get_rew(self, x_velocity: float, action):
    forward_reward = self._forward_reward_weight * x_velocity  # Reward forward movement
    control_cost = self.control_cost(action)  # Penalize for large actions
    healthy_reward = self.healthy_reward  # Reward if hopper is healthy

    # Total reward combines these elements
    reward = forward_reward - control_cost + healthy_reward  

    # Preparing data for logging and debugging
    reward_info = {
        "forward_reward": forward_reward,
        "control_cost": -control_cost,  # Negative as it's a penalty
        "healthy_reward": healthy_reward,
        "total_reward": reward
    }

    return reward, reward_info
