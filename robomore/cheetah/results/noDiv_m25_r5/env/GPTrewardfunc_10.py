def _get_rew(self, x_velocity: float, action):
    # Encourage the cheetah to move forward by rewarding positive x_velocity (moving right)
    forward_reward = self._forward_reward_weight * x_velocity
    
    # Penalize applying too much torque, to promote smoother and more efficient movements
    control_cost = self.control_cost(action)
    
    # Calculate the total reward as the difference between forward_reward and control_cost
    reward = forward_reward - control_cost
    
    # Construct the reward_info dictionary with individual components
    reward_info = {
        'forward_reward': forward_reward,
        'control_cost': control_cost,
    }
    
    return reward, reward_info
