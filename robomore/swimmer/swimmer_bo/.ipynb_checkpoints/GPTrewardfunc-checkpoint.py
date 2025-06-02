import numpy as np
def _get_rew(self, x_velocity: float, action):
    # Encouraging swimmers to sprint intermittently: The goal is to promote short bursts of high-speed swimming
    # followed by periods of lower energy expenditure, which could emulate a sprint-rest behavior.

    # Determine if a burst is in progress or if we are in a resting phase.
    # We will toggle the state based on the x_velocity exceeding a high threshold or falling below a lower threshold.
    high_speed_threshold = 0.8  # Threshold to define a high-speed sprint
    low_speed_threshold = 0.3   # Threshold to define low-energy swimming/rest
    current_speed_high = x_velocity > high_speed_threshold
    current_speed_low = x_velocity < low_speed_threshold

    # State check - use class variables to remember state (assuming they are initialized in other parts of the class)
    if not hasattr(self, 'is_sprinting'):
        self.is_sprinting = False

    # Update the sprinting state based on velocity thresholds
    if current_speed_high:
        self.is_sprinting = True
    elif current_speed_low:
        self.is_sprinting = False

    # Sprint reward: high reward for maintaining sprint speed
    if self.is_sprinting:
        sprint_reward = 2.0 * np.tanh(x_velocity)  # Using tanh to bound the reward
    else:
        sprint_reward = 0

    # Rest reward: modest reward for low speeds to encourage recovery
    if not self.is_sprinting:
        rest_reward = 0.5 * np.tanh(1 - x_velocity)  # Reward lower velocities during rest
    else:
        rest_reward = 0

    # Control cost: penalize large torques to avoid unnecessary flailing and promote efficient motion.
    control_penalty = self.control_cost(action)

    # Compute the total reward from all components
    total_reward = sprint_reward + rest_reward - control_penalty

    # Reward info for tracking individual components
    reward_info = {
        'sprint_reward': sprint_reward,
        'rest_reward': rest_reward,
        'control_penalty': control_penalty,
        'is_sprinting': self.is_sprinting
    }
    print("using the right reawrd")

    return total_reward, reward_info
