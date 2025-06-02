import numpy as np

# 激励弹跳
def _get_rew(self, x_velocity: float, action):
    forward_reward = x_velocity * self._forward_reward_weight

    z_velocity = self.data.qvel[2]
    z_position = self.data.qpos[2]
    # bounce_reward = np.abs(z_velocity) + max(0, z_position - 0.25)
    bounce_reward = z_position 

    healthy_reward = self.healthy_reward

    rewards = forward_reward + healthy_reward + bounce_reward*0.1

    ctrl_cost = self.control_cost(action)
    contact_cost = self.contact_cost
    costs = ctrl_cost + contact_cost

    reward = rewards - costs

    reward_info = {
        "reward_forward": forward_reward,
        "reward_bounce": bounce_reward,
        "reward_ctrl": -ctrl_cost,
        "reward_contact": -contact_cost,
        "reward_survive": healthy_reward,
    }

    return reward, reward_info
