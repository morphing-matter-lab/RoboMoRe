def _get_rew(self, x_velocity: float, action):
    # Encourage the cheetah to move forward rapidly
    forward_reward = self._forward_reward_weight * x_velocity
    
    # Penalize the cheetah for using too much control power
    control_cost = self.control_cost(action)
    
    # Compute the total reward incorporating both forward motion and control cost
    reward = forward_reward - control_cost
    
    # Gather components of the reward for logging and debugging purposes
    reward_info = {
        'forward_reward': forward_reward,
        'control_cost': control_cost,
        'total_reward': reward
    }
    
    return reward, reward_info
