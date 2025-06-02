import numpy as np 
def _get_rew(self, x_velocity: float, action):
    # Reward for vertical position (jump height)
    z_position = self.data.qpos[2]
    jump_reward = np.exp(z_position - 0.5)  # Subtract 0.5 to encourage lifting off

    # Control cost (penalize excessive use of action torques)
    control_cost = self.control_cost(action)

    # Stability reward (encourage the ant to maintain a stable torso orientation)
    torso_quat = self.data.qpos[3:7]  # Quaternion representing the torso orientation
    # A quaternion close to [1, 0, 0, 0] means no rotation from initial position
    # Penalize distance from the identity quaternion to encourage less rotation
    stability_reward = -np.linalg.norm(torso_quat - np.array([1.0, 0.0, 0.0, 0.0]))

    # Compute the total reward
    reward = jump_reward - control_cost + stability_reward

    # Dictionary containing detailed information about reward components
    reward_info = {
        "jump_reward": jump_reward,
        "control_cost": control_cost,
        "stability_reward": stability_reward,
    }

    # Return the total reward and the detailed reward info
    return reward, reward_info
