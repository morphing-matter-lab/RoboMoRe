import numpy as np 
def _get_rew(self, x_velocity: float, action):
    # Define efficient swimming characteristics. Aim to maintain balance between forward motion and control.
    
    # Base forward reward for moving right, placed higher for higher velocities
    forward_reward = self._forward_reward_weight * (x_velocity ** 2)  # Quadratic reward amplifies strong velocities

    # Introduce a reward for stability: Encourage minimal yawing (side ways movements).
    stability_reward = -0.5 * (self.data.qpos[1] ** 2)  # Penalizing lateral position squared to discourage side motion

    # Punishing excessive control inputs, to favor smooth swimming rather than jerky movements
    control_penalty = self.control_cost(action)

    # Incentivize maintaining a certain baseline velocity (e.g., cruising speed), with a soft penalty for deviations
    ideal_velocity = 1.0
    velocity_penalty = 0.5 * ((x_velocity - ideal_velocity) ** 2)  # Penalizes big deviations from ideal

    # Compute the total reward
    total_reward = forward_reward + stability_reward - control_penalty - velocity_penalty

    # Debugging and analysis information containing detailed components of the reward
    reward_info = {
        'forward_reward': forward_reward,
        'stability_reward': stability_reward,
        'control_penalty': control_penalty,
        'velocity_penalty': velocity_penalty,
    }
    
    return total_reward, reward_info
