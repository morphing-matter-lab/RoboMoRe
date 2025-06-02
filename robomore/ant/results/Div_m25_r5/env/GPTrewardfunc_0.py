import numpy as np
def _get_rew(self, x_velocity: float, action):
    # Primary reward for moving forward in the x direction
    forward_reward = self._forward_reward_weight * x_velocity

    # Costs for control usage (motivating efficient use of action torques)
    control_cost = self.control_cost(action)

    # Costs from contact forces (encourage the ant to minimize high force impacts)
    contact_cost = self.contact_cost

    # Combine rewards and costs to form the total reward.
    # Encouraging speed while discouraging excessive use of actuation and minimizing contacts.
    reward = forward_reward - control_cost - contact_cost + self.healthy_reward

    # Information dict to track individual components of the reward function
    reward_info = {
        'forward_reward': forward_reward,
        'control_cost': control_cost,
        'contact_cost': contact_cost,
        'healthy_reward': self.healthy_reward
    }

    return reward, reward_info
