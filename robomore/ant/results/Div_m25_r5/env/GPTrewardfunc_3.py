import numpy as np
def _get_rew(self, x_velocity: float, action):
    # Encourage not just moving forward but also involving some controlled lateral movement 
    # to demonstrate agility and dynamic control capability.
    
    # Reward for moving forward
    forward_reward = self._forward_reward_weight * x_velocity
    
    # Reward for controlled lateral movement (moderate y_velocity to demonstrate lateral agility)
    target_y_velocity = 0.5  # moderate lateral speed target, can be tuned
    lateral_movement_reward = -np.abs(self.data.qvel[1] - target_y_velocity) * self._forward_reward_weight
    
    # Minimize the control effort to promote energy efficiency.
    control_cost = self.control_cost(action)
    
    # Health reward for maintaining a physically feasible and stable posture.
    health_reward = self.healthy_reward
    
    # Total reward computation
    reward = forward_reward + lateral_movement_reward - control_cost + health_reward
    
    # Reward details for debugging and analysis purposes
    reward_info = {
        'forward_reward': forward_reward,
        'lateral_movement_reward': lateral_movement_reward,
        'control_cost': control_cost,
        'health_reward': health_reward
    }

    return reward, reward_info
