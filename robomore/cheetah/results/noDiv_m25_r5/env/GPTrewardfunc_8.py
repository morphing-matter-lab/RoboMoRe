def _get_rew(self, x_velocity: float, action):
    # Reward based on the forward velocity. Positive velocities to the right are encouraged.
    forward_reward = self._forward_reward_weight * x_velocity

    # Control cost penalizes the amount of torque applied to reduce unnecessary flailing.
    control_cost = self.control_cost(action)

    # Calculate the total reward by combining the forward reward and the control cost.
    total_reward = forward_reward - control_cost

    # Organize individual reward components for debugging and inspection.
    reward_info = {
        "forward_reward": forward_reward,
        "control_cost": control_cost,
        "total_reward": total_reward
    }

    return total_reward, reward_info
