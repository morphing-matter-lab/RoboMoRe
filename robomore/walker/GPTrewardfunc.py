import numpy as np
def _get_rew(self, x_velocity: float, action):
    # Reward for forward motion, scaled exponentially to favor higher speeds but with diminishing returns
    forward_reward = np.exp(0.3 * x_velocity) - 1

    # Penalty to discourage excessive use of actuator torques, promoting energy efficiency
    control_penalty = self.control_cost(action)
    
    # Bonus for maintaining a healthy mechanics of motion, which includes staying upright    
    health_bonus = self.healthy_reward
    
    # Additional reward for synchronous and rhythmic leg movements
    # We can leverage the sine of the sum of relevant joint angles to promote a smooth cyclic locomotion
    angles = self.data.qpos[2:7]  # Assuming indices 2-6 are joint angles
    rhythmic_movement_bonus = np.sum(np.sin(angles))

    # Compute total reward considering all the components
    reward = forward_reward + rhythmic_movement_bonus + health_bonus - control_penalty
    
    # Organize detailed reward components for diagnostic purposes
    reward_info = {
        'forward_reward': forward_reward,
        'rhythmic_movement_bonus': rhythmic_movement_bonus,
        'health_bonus': health_bonus,
        'control_penalty': control_penalty
    }
    
    return reward, reward_info
