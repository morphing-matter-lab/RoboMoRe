def _get_rew(self, x_velocity: float, action):
    # Reward for moving forward: scaled by a defined weight
    forward_reward = self._forward_reward_weight * x_velocity
    
    # Cost for using controls (applying torque), helps keep the control output smooth and bounded
    control_cost = self.control_cost(action)
    
    # Total reward: encourage speed but discourage excessive control effort
    reward = forward_reward - control_cost
    
    # Detailed reward information for debugging and analysis
    reward_info = {
        "forward_reward": forward_reward,
        "control_cost": control_cost,
        "net_reward": reward
    }
    
    return reward, reward_info
