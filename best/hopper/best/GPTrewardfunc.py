import numpy as np
def _get_rew(self, x_velocity: float, action):
    # Encourage efficient forward movement by not just rewarding speed but also smooth progression
    smoothness_reward = -np.sum(np.abs(np.diff(action)))  # Decrease reward for large differences in consecutive actions

    # Reward forward velocity, with an exponential component to prioritize higher speeds exponentially
    exponential_speed_reward = self._forward_reward_weight * np.exp(x_velocity) - 1  # Using exp to exponentially prefer higher velocities, subtract 1 to center around zero for small velocities

    # Penalty for using too much control input, which promotes efficiency
    control_penalty = self._ctrl_cost_weight * np.sum(np.square(action))

    # Healthy state reward, keeping the hopper upright and in a healthy range
    health_bonus = self.healthy_reward

    # Total reward calculation
    total_reward = exponential_speed_reward + smoothness_reward - control_penalty + health_bonus

    # Tracking reward details for better understanding and debugging
    reward_info = {
        'smoothness_reward': smoothness_reward,
        'exponential_speed_reward': exponential_speed_reward,
        'control_penalty': control_penalty,
        'health_bonus': health_bonus,
        'total_reward': total_reward
    }

    return total_reward, reward_info
