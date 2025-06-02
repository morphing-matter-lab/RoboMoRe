def _get_rew(self, x_velocity: float, action):
    # Rewarding forward movement (in the x-direction)
    forward_reward = self._forward_reward_weight * x_velocity

    # Penalizing the control effort to encourage efficient movements
    control_cost = self._ctrl_cost_weight * np.sum(np.square(action))

    # Penalizing for large contact forces to encourage smooth locomotion
    contact_cost = self._contact_cost_weight * np.sum(np.square(self.contact_forces))

    # Providing a reward for maintaining a healthy state
    healthy_reward = self.healthy_reward

    # Calculate the total reward
    reward = forward_reward + healthy_reward - control_cost - contact_cost

    # Information breakdown for analysis
    reward_info = {
        'forward_reward': forward_reward,
        "control_cost": control_cost,
        "contact_cost": contact_cost,
        "healthy_reward": healthy_reward
    }

    return reward, reward_info
