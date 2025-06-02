import numpy as np 
def _get_rew(self, x_velocity: float, action):
    # Reward for moving forward, positive proportional to the velocity towards the right
    forward_reward = self._forward_reward_weight * x_velocity

    # Control cost to penalize too much action use (energy spent)
    ctrl_cost = self.control_cost(action)

    # Calculate the overall reward: encourage speed but penalize excessive control usage
    reward = forward_reward - ctrl_cost
    
    # Optionally, you could experiment with exponential scaling to emphasize forward motion
    # reward = np.exp(forward_reward / 10) - np.sqrt(ctrl_cost)

    # Prepare reward components dictionary for detailed information
    reward_info = {
        "reward_forward": forward_reward,
        "reward_ctrl": -ctrl_cost,
    }

    return reward, reward_info
