import numpy as np 
def _get_rew(self, x_velocity: float, action):
    # Reward is proportional to the forward velocity but penalizes the cost of control to prevent erratic movements.
    forward_reward = self._forward_reward_weight * x_velocity  # Encourages forward motion.
    ctrl_cost = self.control_cost(action)  # Penalizes excessive control signals.

    # You can experiment with different reward formulations here.
    # For example, using exponential function to emphasize faster speeds more significantly:
    # forward_reward = self._forward_reward_weight * np.exp(x_velocity)

    reward = forward_reward - ctrl_cost  # Total reward calculation.

    # Additional information for debugging and analysis; 
    # breaks down contribution of velocity and control cost to the overall reward.
    reward_info = {
        "reward_forward": forward_reward,
        "reward_ctrl": -ctrl_cost,
    }

    return reward, reward_info
