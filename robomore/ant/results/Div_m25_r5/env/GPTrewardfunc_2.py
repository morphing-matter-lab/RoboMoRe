import numpy as np
def _get_rew(self, x_velocity: float, action):
    # Encouraging fast, straight-line forward movement by maximizing x_velocity and 
    # penalizing deviation from the forward path (y_velocity should remain close to 0).
    forward_reward = self._forward_reward_weight * x_velocity
    straight_line_penalty = -abs(self.data.qvel[1])  # Penalizing y_velocity

    # Encourage efficient utilization of actions by minimizing the cumulative use of joint torques.
    control_cost = self.control_cost(action)

    # Incentivize maintaining a healthy robot posture, directly linked to fitness and robot integrity.
    health_reward = self.healthy_reward

    # Incorporate a term to reward stability. Here, the agent earns additional points by maintaining
    # minimal variation in neural vertical height (z-position), promoting stability.
    z_position = self.data.qpos[2]  # Vertical position of the main body
    target_z_position = 0.7  # Target steady state height
    stability_reward = -np.square(z_position - target_z_position)

    # Total reward calculation
    reward = forward_reward + straight_line_penalty + stability_reward - control_cost + health_reward

    # Reward information dictionary for introspection and debugging
    reward_info = {
        'forward_reward': forward_reward,
        'straight_line_penalty': straight_line_penalty,
        'stability_reward': stability_reward,
        'control_cost': control_cost,
        'health_reward': health_reward
    }

    return reward, reward_info
