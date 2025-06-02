def _get_rew(self, x_velocity: float, action):
    # Reward for moving forward. The faster, the better.
    forward_reward = self._forward_reward_weight * x_velocity

    # Control cost penalizes the magnitude of the action (torque) to avoid excessive flailing.
    control_cost = self.control_cost(action)

    # Total reward calculation
    reward = forward_reward - control_cost

    # Reward breakdown information
    reward_info = {
        "forward_reward": forward_reward,
        "control_cost": control_cost,
        "net_reward": reward
    }

    return reward, reward_info
