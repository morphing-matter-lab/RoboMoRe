def _get_rew(self, x_velocity: float, action):
    # Rewarding forward movement
    forward_reward = self._forward_reward_weight * x_velocity

    # Penalizing the cost of control/actions (torque applied)
    control_cost = self.control_cost(action)
    
    # Total reward calculation
    reward = forward_reward - control_cost

    # Information about individual components of the reward
    reward_info = {
        "forward_reward": forward_reward,
        "control_cost": control_cost,
    }

    return reward, reward_info
