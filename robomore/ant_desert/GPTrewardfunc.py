import numpy as np 
def _get_rew(self, x_velocity: float, action):
    # Reward for forward movement, emphasizing an exponential growth to encourage high speeds.
    forward_reward = self._forward_reward_weight * np.exp(x_velocity - 1)  # Exponential growth for incentivizing high speeds

    # Encourage lateral stability: penalize the absolute value of lateral (y-axis) velocity
    y_velocity = self.data.qvel[1]
    lateral_stability_penalty = -np.abs(y_velocity)  # Negative as we want to minimize lateral movement

    # Control cost to penalize excessive energy usage
    control_cost = self.control_cost(action)

    # Keep the robot in a healthy state
    healthy_reward = self.healthy_reward

    # Minimize contact costs to promote gentle contacts with the ground
    contact_cost = self.contact_cost

    # Total reward calculation combining all components
    reward = forward_reward + healthy_reward + lateral_stability_penalty - control_cost - contact_cost

    # Detailed reward breakdown for analysis and debugging
    reward_info = {
        'forward_reward': forward_reward,
        'lateral_stability_penalty': lateral_stability_penalty,
        'control_cost': control_cost,
        'contact_cost': contact_cost,
        'healthy_reward': healthy_reward
    }

    return reward, reward_info
