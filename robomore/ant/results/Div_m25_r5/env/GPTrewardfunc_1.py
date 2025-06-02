import numpy as np
def _get_rew(self, x_velocity: float, action):
    # Reward for efficient forward motion with minimization of sideways drift (y_velocity close to 0)
    forward_efficiency_reward = self._forward_reward_weight * x_velocity
    sideways_drift_penalty = -self._forward_reward_weight * abs(self.data.qvel[1])  # Penalty for y_velocity

    # Penalize for oscillations in movement - Oscillations can be inferred by monitoring large changes in x velocity
    velocity_smoothness_penalty = -self._forward_reward_weight * np.abs(self.data.qvel[3])  # Change in x velocity over time

    # Control cost: Encourage the agent to use less energy in movements
    control_cost = self.control_cost(action)

    # Health reward: Encourage the agent to maintain a physically feasible posture
    health_reward = self.healthy_reward

    # Total reward computation
    reward = (forward_efficiency_reward + sideways_drift_penalty + velocity_smoothness_penalty
              - control_cost + health_reward)

    # Dictionary of all reward components for debugging and analysis in training logs
    reward_info = {
        'forward_efficiency_reward': forward_efficiency_reward,
        'sideways_drift_penalty': sideways_drift_penalty,
        'velocity_smoothness_penalty': velocity_smoothness_penalty,
        'control_cost': control_cost,
        'health_reward': health_reward
    }

    return reward, reward_info
