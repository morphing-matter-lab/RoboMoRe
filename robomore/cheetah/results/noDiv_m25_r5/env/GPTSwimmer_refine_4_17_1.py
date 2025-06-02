import numpy as np 
def _get_rew(self, x_velocity: float, action):
    # Reward for moving forward.
    forward_reward = self._forward_reward_weight * x_velocity

    # Control cost penalizes the magnitude of action (torque).
    ctrl_cost = self.control_cost(action)

    # Optional: Penalize instability by influencing the reward with angular velocities.
    # Not forcefully implemented here but can be considered for enhancing the learning stability.
    # angular_vel_cost = 0.1 * np.sum(np.square(self.data.qvel[1:]))

    # Compute the total reward: reward for traveling forward minus the control cost.
    # Including optional penalties like angular velocity would look like this:
    # reward = forward_reward - ctrl_cost - angular_vel_cost
    reward = forward_reward - ctrl_cost

    # Additional reward components for detailed logging or monitoring.
    reward_info = {
        "reward_forward": forward_reward,
        "reward_ctrl": -ctrl_cost,
        # "reward_stability": -angular_vel_cost  # Uncomment if using angular velocity cost
    }

    return reward, reward_info
