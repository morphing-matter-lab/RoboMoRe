def _get_rew(self, x_velocity: float, action):
    # Reward for moving forwards, more reward for faster x velocity    
    forward_reward = self._forward_reward_weight * x_velocity  

    # Cost of using the actuators, penalizes excessive use of energy
    # Sum of squares of action components scaled by control cost weight
    control_penalty = self._ctrl_cost_weight * np.sum(np.square(action))

    # Total reward is the forward movement reward minus the control cost
    total_reward = forward_reward - control_penalty

    # Reward info containing the detailed components for debugging/monitoring
    reward_info = {
        'forward_reward': forward_reward,
        'control_penalty': control_penalty
    }

    return total_reward, reward_info
