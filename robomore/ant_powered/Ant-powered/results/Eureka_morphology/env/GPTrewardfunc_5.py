import numpy as np
def _get_rew(self, x_velocity: float, action):
    forward_reward = x_velocity * self._forward_reward_weight
    healthy_reward = self.healthy_reward
    rewards = forward_reward + healthy_reward

    ctrl_cost = self.control_cost(action)
    contact_cost = self.contact_cost
    costs = ctrl_cost + contact_cost

    reward = rewards - costs

    reward_info = {
        "reward_forward": forward_reward,
        "reward_ctrl": -ctrl_cost,
        "reward_contact": -contact_cost,
        "reward_survive": healthy_reward,
    }

    return reward, reward_info    
