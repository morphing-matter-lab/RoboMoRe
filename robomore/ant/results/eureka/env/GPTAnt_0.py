def _get_rew(self, x_velocity: float, action):
    # Reward for forward movement, heavily weigh this to encourage faster movement
    forward_reward = self._forward_reward_weight * x_velocity

    # Penalty for using too much control power (control cost)
    control_cost = self.control_cost(action)

    # Penalty for making contact with the ground too harshly (contact cost)
    contact_cost = self.contact_cost

    # Check if the agent maintains a healthy posture
    healthy_reward = self.healthy_reward

    # Total reward calculation
    reward = forward_reward + healthy_reward - control_cost - contact_cost

    # Reward information for debugging purposes
    reward_info = {
        "forward_reward": forward_reward,
        "control_cost": control_cost,
        "contact_cost": contact_cost,
        "healthy_reward": healthy_reward
    }

    return reward, reward_info
