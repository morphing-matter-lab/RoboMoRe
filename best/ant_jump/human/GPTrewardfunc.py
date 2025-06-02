import numpy as np
def _get_rew(self, x_velocity: float, action):

    z_velocity = self.data.qvel[2]
    bounce_reward = z_velocity
    healthy_reward = self.healthy_reward
    rewards = healthy_reward + bounce_reward
    ctrl_cost = self.control_cost(action)
    contact_cost = self.contact_cost
    costs = ctrl_cost + contact_cost

    reward = rewards - costs

    reward_info = {
        "reward_bounce": bounce_reward,
        "reward_ctrl": -ctrl_cost,
        "reward_contact": -contact_cost,
        "reward_survive": healthy_reward,
    }

    return reward, reward_info