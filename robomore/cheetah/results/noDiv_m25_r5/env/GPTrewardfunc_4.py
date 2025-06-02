def _get_rew(self, x_velocity: float, action):
    # Reward for moving forward, the faster the better
    forward_reward = self._forward_reward_weight * x_velocity

    # Cost of using controls (i.e., applying torque), to discourage unnecessary movement
    control_cost = self.control_cost(action)
    
    # Calculate total reward by combining forward reward and control cost
    reward = forward_reward - control_cost
    
    # Prepare reward information dictionary for detailed tracking
    reward_info = {
        'forward_reward': forward_reward,
        'control_cost': control_cost,
        'total_reward': reward
    }
    
    return reward, reward_info
