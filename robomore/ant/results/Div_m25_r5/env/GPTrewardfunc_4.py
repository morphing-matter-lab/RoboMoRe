import numpy as np
def _get_rew(self, x_velocity: float, action):
    # Encourage forward motion while rewarding periodic lateral movements which simulates a zig-zag or weaving motion
    forward_reward = self._forward_reward_weight * x_velocity
    
    # Enhance the specific motion behavior by rewarding periodically oscillating y-velocities (side-to-side movement)
    desired_y_velocity_frequency = 2 * np.pi * 0.5  # 0.5 Hz as desired oscillation frequency for y-velocity
    current_time = self.sim.data.time
    target_y_velocity = np.sin(desired_y_velocity_frequency * current_time)
    lateral_weaving_reward = -np.square(self.data.qvel[1] - target_y_velocity) * self._forward_reward_weight

    # Penalize high control effort to ensure efficient use of energy and actuation
    control_cost = self.control_cost(action)
    
    # Include a health reward to ensure the agent maintains a viable robot configuration
    health_reward = self.healthy_reward
    
    # Total reward combines forward movement, lateral weaving, control efficiency, and health status
    reward = forward_reward + lateral_weaving_reward - control_cost + health_reward
    
    # Detailed reward information for monitoring and debugging purposes
    reward_info = {
        'forward_reward': forward_reward,
        'lateral_weaving_reward': lateral_weaving_reward,
        'control_cost': control_cost,
        'health_reward': health_reward
    }

    return reward, reward_info
