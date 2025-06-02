def _get_rew(self, x_velocity: float, action):
    # Forward movement reward: This encourages the agent to move as fast as possible in the x-direction.
    forward_reward = self._forward_reward_weight * x_velocity

    # Control cost reward: This penalizes the agent for using too much force/torque, encouraging energy efficiency.
    control_cost = self.control_cost(action)

    # Contact cost reward: This penalizes the agent for making excessive contact with the ground.
    contact_cost = self.contact_cost

    # Health bonus: This provides a constant reward as long as the ant remains healthy.
    healthy_bonus = self.healthy_reward

    # Total reward computation: This combines all the components.
    reward = forward_reward - control_cost - contact_cost + healthy_bonus

    # Reward info dictionary: This helps in debugging by showing the contribution of each component to the total reward.
    reward_info = {
        'forward_reward': forward_reward,
        'control_cost': control_cost,
        'contact_cost': contact_cost,
        'healthy_bonus': healthy_bonus
    }

    return reward, reward_info
