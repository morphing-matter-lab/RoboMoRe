import numpy as np
def _get_rew(self, x_velocity: float, action):
    # Encouraging a new motion behavior: emphasize zigzagging while moving forward
    # This could model situations in real-world scenarios where agility and directional changes are key.

    # Simple heuristic for zigzag: use the absolute sine of yaw
    # Assuming the walker has a yaw component in qpos that reflects overall orientation
    yaw_index = 3  # Hypothetical index of yaw in qpos if available
    zigzag_bonus = np.abs(np.sin(self.data.qpos[yaw_index]))

    # Forward speed reward with less emphasis as compared to typical straight reward to integrate zigzag importance
    forward_vel_reward = 0.5 * np.clip(x_velocity, 0, 3)  # Clipping to max speed 3 to prevent too high a reward
    
    # Penalizing high control costs to maintain energy efficiency especially when changing directions
    control_penalty = self.control_cost(action)

    # Continue to maintain health state rewards to ensure the walker doesn't fall over
    health_bonus = self.healthy_reward
    
    # Combine all the components into a total reward function
    reward = forward_vel_reward + 5 * zigzag_bonus + health_bonus - control_penalty
    
    # Detailed breakout for easier troubleshooting and tweaking
    reward_info = {
        'forward_vel_reward': forward_vel_reward,
        'zigzag_bonus': zigzag_bonus,
        'health_bonus': health_bonus,
        'control_penalty': control_penalty
    }
    
    return reward, reward_info
