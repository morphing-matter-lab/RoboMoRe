
from gymnasium.envs.registration import register

register(
    id="GPTAntEnv",  
    entry_point="GPTAnt:GPTAntEnv",  
    max_episode_steps=1000,  
    reward_threshold=5000.0,  
)

import gymnasium as gym
from gymnasium.envs.registration import register

def register_gpt_ant():
    env_id = "GPTAntEnv"  # 你的环境 ID

    # 🔹 先检查环境是否已注册，若存在，则移除
    if env_id in gym.registry:
        del gym.registry[env_id]  # Gymnasium 可能已更新 registry 结构

    # 🔹 重新注册环境，确保 Gym 载入最新的 `GPTAnt.py`
    register(
        id=env_id,
        entry_point="GPTAnt:GPTAntEnv",  # 确保 `GPTAnt.py` 里有 `GPTAntEnv` 类
        max_episode_steps=1000,
        reward_threshold=5000.0,
    )

    print(f"✅ 重新注册环境: {env_id}")
