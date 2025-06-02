def _get_rew(self, x_velocity: float, action):
    # Reward for moving forward
    forward_reward = self._forward_reward_weight * x_velocity
    
    # Cost for using controls (torque)
    control_cost = self.control_cost(action)
    
    # Total reward is the forward movement reward minus the control cost
    reward = forward_reward - control_cost
    
    # Information about the reward components
    reward_info = {
        'forward_reward': forward_reward,
        'control_cost': control_cost,
        'total_reward': reward
    }
    
    return reward, reward_info
