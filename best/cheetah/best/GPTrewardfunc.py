import numpy as np 
def _get_rew(self, x_velocity: float, action):
    # Reward for moving forward emphasizing higher speeds
    forward_reward = self._forward_reward_weight * x_velocity

    # Calculate control cost using the predefined method
    control_cost = self.control_cost(action)

    # Reward for energy efficiency: velocity per control effort
    efficiency = x_velocity / (control_cost + 1e-5)  # Avoid division by zero
    normalized_efficiency_reward = np.exp(efficiency) - 1  # Shifted by -1 to normalize around 0

    # Calculate smoothness reward: penalize fluctuations in velocity
    if not hasattr(self, 'prev_velocity'):
        self.prev_velocity = x_velocity
    smoothness_penalty = -np.abs(x_velocity - self.prev_velocity)  # Penalize changes in velocity
    self.prev_velocity = x_velocity  # Update the previous velocity for the next step
    smoothness_reward = np.exp(smoothness_penalty) - 1  # Normalize the smoothness reward
    
    # Action symmetry bonus: rewards symmetrical actions between limbs
    if len(action) % 2 == 0:  
        left_actions = action[1::2]  
        right_actions = action[0::2]  
        symmetry_penalty = -np.sum(np.abs(left_actions - right_actions))  
    else:
        symmetry_penalty = 0  
    symmetry_reward = np.exp(symmetry_penalty) - 1  
    
    # Combine all components to form the total reward
    total_reward = forward_reward - control_cost + normalized_efficiency_reward + smoothness_reward + symmetry_reward

    # Reward info dictionary for debugging and analysis
    reward_info = {
        'forward_reward': forward_reward,
        'control_cost': control_cost,
        'normalized_efficiency_reward': normalized_efficiency_reward,
        'smoothness_reward': smoothness_reward,
        'symmetry_reward': symmetry_reward,
        'total_reward': total_reward
    }

    return total_reward, reward_info
