import numpy as np 
def _get_rew(self, x_velocity: float, action):
    # Encourage forward movement with a target velocity for optimal speed
    target_velocity = 1.0  # Desired forward velocity
    velocity_error = np.abs(x_velocity - target_velocity)  # Calculate deviation from target
    forward_reward = self._forward_reward_weight * np.exp(-velocity_error)  # Exponential decay for reward

    # Penalize lateral (y-direction) movement to maintain focus on forward motion
    y_velocity_penalty = -abs(self.data.qvel[1])  # Penalize any motion in the y direction

    # Introduce a reward for efficient alternate leg movement to promote stability and locomotion style
    leg_pair_1 = np.abs(action[0] + action[1] - (action[4] + action[5]))
    leg_pair_2 = np.abs(action[2] + action[3] - (action[6] + action[7]))
    alternate_leg_movement_reward = (np.exp(-leg_pair_1) + np.exp(-leg_pair_2)) / 2.0

    # Maintain healthy posture by rewarding stability within the Z-limits
    health_reward = self.healthy_reward

    # Evaluate the efficiency of movement through control costs
    control_cost = self.control_cost(action)
    
    # Penalize excessive contact forces to minimize instability 
    contact_cost = self.contact_cost

    # Calculate total reward by combining all components
    reward = forward_reward + y_velocity_penalty + alternate_leg_movement_reward - control_cost + health_reward - contact_cost

    # Reward info for monitoring individual components
    reward_info = {
        "forward_reward": forward_reward,
        "y_velocity_penalty": y_velocity_penalty,
        "alternate_leg_movement_reward": alternate_leg_movement_reward,
        "control_cost": control_cost,
        "contact_cost": contact_cost,
        "health_reward": health_reward,
    }

    return reward, reward_info
