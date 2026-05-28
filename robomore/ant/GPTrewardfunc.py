import numpy as np
def _get_rew(self, x_velocity: float, action):
    # Fitness-style reward:
    # 1) a forward-speed "milestone ladder" that keeps rewarding faster running,
    # 2) a soft-OR bank of different leg-motion patterns so multiple gaits can score well,
    # 3) entropy/richness terms to encourage all legs and contacts to participate,
    # 4) only a small penalty for severe action saturation, not ordinary torque use.

    alive = 1.0 if self.is_healthy else 0.0

    # --- Forward speed ladder: bounded bonuses for passing speed thresholds plus an open-ended tail.
    thresholds = np.array([0.15, 0.45, 0.80, 1.20, 1.70, 2.30, 3.00])
    weights = np.array([0.15, 0.20, 0.30, 0.45, 0.65, 0.90, 1.20])
    speed_steps = 1.0 / (1.0 + np.exp(-(x_velocity - thresholds) / 0.12))
    speed_ladder_reward = np.sum(weights * speed_steps)

    high_speed_tail = 0.60 * np.log1p(np.exp(x_velocity - 2.5))
    backward_penalty = 1.25 / (1.0 + np.exp((x_velocity + 0.05) / 0.10))

    # --- Joint/leg motion extraction.
    joint_pos = self.data.qpos[-8:]
    joint_vel = self.data.qvel[-8:]
    leg_vel = joint_vel.reshape(4, 2)

    leg_energy = np.sqrt(np.sum(np.square(leg_vel), axis=1) + 1e-8)
    mean_leg_energy = np.mean(leg_energy)
    activity = np.tanh(mean_leg_energy / 3.0)

    # --- Soft gait portfolio: trot, pace, bound, pronk-like patterns are all accepted.
    # This encourages many possible motion behaviours instead of prescribing one gait.
    leg_signal = np.tanh(0.65 * leg_vel[:, 0] + 0.35 * leg_vel[:, 1])
    signal_norm = np.linalg.norm(leg_signal) + 1e-6

    gait_templates = np.array([
        [1.0, -1.0, -1.0, 1.0],   # diagonal / trot-like
        [1.0, 1.0, -1.0, -1.0],   # left-right / pace-like
        [1.0, -1.0, 1.0, -1.0],   # front-back / bound-like
        [1.0, 1.0, 1.0, 1.0],     # pronk / synchronous
    ])

    gait_corr = np.abs(np.dot(gait_templates, leg_signal) / (2.0 * signal_norm))
    gait_corr = np.clip(gait_corr, 0.0, 1.0)

    sharpness = 5.0
    max_arg = sharpness * gait_corr
    max_val = np.max(max_arg)
    soft_best_gait = (max_val + np.log(np.mean(np.exp(max_arg - max_val)))) / sharpness

    gait_portfolio_reward = 0.85 * activity * soft_best_gait

    # --- Encourage distributed leg use rather than dragging or relying on one limb.
    leg_prob = leg_energy / (np.sum(leg_energy) + 1e-8)
    leg_entropy = -np.sum(leg_prob * np.log(leg_prob + 1e-8)) / np.log(4.0)
    leg_usage_reward = 0.45 * activity * leg_entropy

    # --- Encourage expressive joint excursions, but saturate so extreme contortions do not dominate.
    joint_excursion = np.mean(np.abs(np.sin(joint_pos)))
    joint_expression_reward = 0.25 * activity * joint_excursion

    # --- Contact participation diversity: useful ground interaction without directly rewarding impact size.
    contact_norms = np.linalg.norm(self.contact_forces[1:], axis=1)
    contact_sum = np.sum(contact_norms)
    contact_prob = contact_norms / (contact_sum + 1e-8)
    contact_entropy = -np.sum(contact_prob * np.log(contact_prob + 1e-8)) / np.log(contact_prob.size + 1e-8)
    contact_presence = np.tanh(np.mean(contact_norms))
    contact_diversity_reward = 0.25 * contact_presence * contact_entropy

    # --- Small dynamic-body-motion bonus to avoid overly static shuffling.
    root_vertical_speed = np.abs(self.data.qvel[2])
    root_roll_pitch_speed = np.linalg.norm(self.data.qvel[3:5])
    body_dynamics_reward = 0.15 * np.tanh(root_vertical_speed + 0.25 * root_roll_pitch_speed)

    # --- Mild penalty only for pushing actions close to saturation.
    saturation = np.maximum(np.abs(action) - 0.85, 0.0)
    action_saturation_penalty = 0.20 * np.mean(np.square(saturation))

    reward = alive * (
        speed_ladder_reward
        + high_speed_tail
        + gait_portfolio_reward
        + leg_usage_reward
        + joint_expression_reward
        + contact_diversity_reward
        + body_dynamics_reward
        - backward_penalty
        - action_saturation_penalty
        + 0.20
    ) - (1.0 - alive) * 2.0

    reward_info = {
        "reward_speed_ladder": speed_ladder_reward,
        "reward_high_speed_tail": high_speed_tail,
        "reward_backward_penalty": -backward_penalty,
        "reward_gait_portfolio": gait_portfolio_reward,
        "reward_leg_usage_entropy": leg_usage_reward,
        "reward_joint_expression": joint_expression_reward,
        "reward_contact_diversity": contact_diversity_reward,
        "reward_body_dynamics": body_dynamics_reward,
        "reward_action_saturation": -action_saturation_penalty,
        "reward_alive": alive,
        "x_velocity": x_velocity,
    }

    return reward, reward_info
