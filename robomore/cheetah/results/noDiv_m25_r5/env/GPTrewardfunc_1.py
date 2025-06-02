def _get_rew(self, x_velocity: float, action):
    # Reward for moving forwards
    forward_reward = self._forward_reward_weight * x_velocity
    
    # Penalize the control effort to encourage efficient movements
    control_cost = self.control_cost(action)
    
    # Total reward combining the forward movement and control costs
    reward = forward_reward - control_cost
    
    # Dictionary to pass additional reward information for debugging or analysis
    reward_info = {
        "forward_reward": forward_reward,
        "control_cost": control_cost,
        "net_reward": reward
    }
    
    return reward, reward_info
