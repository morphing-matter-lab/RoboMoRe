def _get_rew(self, x_velocity: float, action):
    # Reward for moving forward: directly related to the velocity along the x-axis
    forward_reward = self._forward_reward_weight * x_velocity

    # Penalty for the use of control inputs (torque), to promote energy efficiency
    control_penalty = self.control_cost(action)
    
    # Total reward combines moving forward and control penalty
    reward = forward_reward - control_penalty
    
    # Dictionary containing detailed components of the reward for debugging and analysis
    reward_info = {
        "forward_reward": forward_reward,
        "control_penalty": control_penalty
    }

    return reward, reward_info
