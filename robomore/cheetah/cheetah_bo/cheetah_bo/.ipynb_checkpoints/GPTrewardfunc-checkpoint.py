import numpy as np
def _get_rew(self, x_velocity: float, action):
    # Reward function to encourage high-speed leap-and-recover style of motion.
    # The idea is to reward significant positive changes in velocity, simulating a leaping motion,
    # followed by minimal action to maintain a pose, simulating recovery. This mimics explosive
    # power actions followed by stability phases common in various dynamic sports or animal movements.

    # Calculate basic forward motion reward
    forward_reward = self._forward_reward_weight * x_velocity

    # Calculate control cost
    control_cost = self.control_cost(action)

    # Encourage big positive spikes in velocity (leaping) followed by periods of lower velocity (recovery)
    # By rewarding the derivative of the velocity (acceleration) when it's positive and above a threshold
    if not hasattr(self, 'last_velocity'):
        self.last_velocity = x_velocity
    velocity_change = x_velocity - self.last_velocity
    self.last_velocity = x_velocity

    # Only reward positive spikes in velocity change which are greater than 0.1 (threshold for 'leap')
    leap_reward = 0
    if velocity_change > 0.1:
        leap_reward = np.exp(velocity_change * 10) - 1  # Scale and exponentiate for significant impact

    # Combine all components
    total_reward = forward_reward - control_cost + leap_reward

    # Reward info for debugging purposes
    reward_info = {
        'forward_reward': forward_reward,
        'control_cost': control_cost,
        'leap_reward': leap_reward,
        'total_reward': total_reward
    }

    return total_reward, reward_info
