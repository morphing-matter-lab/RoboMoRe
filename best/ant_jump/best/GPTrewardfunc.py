import numpy as np 
def _get_rew(self, x_velocity: float, action):
    # Decomposing the reward function
    jump_height = self.data.body(self._main_body).xpos[2]  # Z position gives the height
    
    # Reward for jumping higher
    height_reward = np.exp(jump_height - 1)  # exponential reward starting from height 1
    
    # Control cost to make sure the robot uses minimum effort to jump
    control_cost = self.control_cost(action)
    
    # Contact cost to penalize excessive force usage in contacts, promoting smooth jumping
    contact_cost = self.contact_cost
    
    # Component to support healthy posture
    healthy_posture_reward = self.healthy_reward
    
    # Combination of different components of the reward
    reward = (
        height_reward * self._forward_reward_weight 
        - contact_cost 
        - control_cost 
        + healthy_posture_reward
    )
    
    # Reward info for better analysis and debugging
    reward_info = {
        "height_reward": height_reward,
        "control_cost": control_cost,
        "contact_cost": contact_cost,
        "healthy_posture_reward": healthy_posture_reward,
    }
    
    return reward, reward_info
