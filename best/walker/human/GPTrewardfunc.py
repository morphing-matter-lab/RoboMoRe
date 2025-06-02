import numpy as np

def _get_rew(self, x_velocity: float, action):
    forward_reward = self._forward_reward_weight * x_velocity
    healthy_reward = self.healthy_reward
    rewards = forward_reward + healthy_reward

    ctrl_cost = self.control_cost(action)
    costs = ctrl_cost
    reward = rewards - costs

    reward_info = {
        "reward_forward": forward_reward,
        "reward_ctrl": -ctrl_cost,
        "reward_survive": healthy_reward,
    }

    return reward, reward_info