import numpy as np

def _get_rew(self, x_velocity: float, action):
    forward_reward = self._forward_reward_weight * x_velocity
    ctrl_cost = self.control_cost(action)

    reward = forward_reward - ctrl_cost

    reward_info = {
        "reward_forward": forward_reward,
        "reward_ctrl": -ctrl_cost,
    }
    return reward, reward_info