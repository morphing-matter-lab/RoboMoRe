import numpy as np 
def _get_rew(self, x_velocity: float, action):
    # Forward velocity reward: promoting the agent to move as fast as possible in the x-direction
    forward_reward = self._forward_reward_weight * x_velocity

    # Control cost: penalizes the magnitude of action inputs to stabilize movements
    control_cost = self.control_cost(action)

    # Contact cost: penalizes the robot if the contact forces are too large, indicating potentially harmful interactions with the environment
    contact_cost = self.contact_cost

    # Health bonus: reward for keeping the robot within healthy operating conditions
    healthy_bonus = self.healthy_reward

    # Compute the total reward
    reward = forward_reward + healthy_bonus - control_cost - contact_cost

    # Information dictionary for debugging or additional insights
    reward_info = {
        "forward_reward": forward_reward,
        "control_cost": control_cost,
        "contact_cost": contact_cost,
        "healthy_bonus": healthy_bonus,
    }

    return reward, reward_info
