import os
import logging
from datetime import datetime
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import time
import gymnasium as gym
import sys
import numpy as np
from gymnasium.wrappers import RecordVideo
from design import *
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
import torch
import math
from stable_baselines3.common.callbacks import BaseCallback
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # 确保 Python 能找到当前目录

class DynamicRewardLoggingCallback(BaseCallback):
    def __init__(self, reward_prefix="reward/", verbose=0):
        super().__init__(verbose)
        self.reward_prefix = reward_prefix

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        for info in infos:
            for key, value in info.items():
                # 自动识别所有 reward 开头的 key
                if key.startswith(self.reward_prefix):
                    self.logger.record(key, value)
        return True
    
    
def setup_logging(div_flag):
    # 创建 results 文件夹（如果不存在）
    results_folder = "results"
    # os.makedirs(results_folder, exist_ok=True)

    # 生成当前时间作为子文件夹名称
    if div_flag:
        folder_name = os.path.join(results_folder, "div" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    else:
        folder_name = os.path.join(results_folder, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    os.makedirs(folder_name, exist_ok=True)

    # 在 `folder_name` 目录下创建 `assets/` 和 `env/` 子文件夹
    assets_folder = os.path.join(folder_name, "assets")
    os.makedirs(assets_folder, exist_ok=True)

    env_folder = os.path.join(folder_name, "env")
    os.makedirs(env_folder, exist_ok=True)

    # 创建日志文件路径
    log_file = os.path.join(folder_name, "parameters.log")

    # 配置 logging
    logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(message)s")
    
    return folder_name  # 返回创建的文件夹路径


def make_env():
    """ 兼容 `SubprocVecEnv` 的 Gym 环境创建函数 """
    def _init():
        current_dir = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(current_dir,"GPTHopper.xml")
        # env = gym.make("Hopper-v5")  # 不能加 render_mode，否则 SubprocVecEnv 可能报错
        env = gym.make("GPTHopperEnv", xml_file = xml_path)
        return env
    return _init


def Train(morphology, rewardfunc, folder_name, total_timesteps = 5e5, stage = 'coarse', callback=False, iter=0):
    num_envs = 16  # 并行环境数量
    # envs = SubprocVecEnv([make_env() for _ in range(num_envs)])
    envs = DummyVecEnv([make_env() for _ in range(num_envs)])
    envs = VecMonitor(envs)  
    
    
    
    if iter:
        model_path = os.path.join(folder_name, f"{stage}", f"SAC_iter{iter}_morphology{morphology}_rewardfunc{rewardfunc}_{total_timesteps}steps")
        tensorboard_name = folder_name+f"/sac_morphology{morphology}_rewardfunc{rewardfunc}_{iter}"
        
    else:
        model_path = os.path.join(folder_name, f"{stage}", f"SAC_morphology{morphology}_rewardfunc{rewardfunc}_{total_timesteps}steps")
        tensorboard_name = folder_name+f"/sac_morphology{morphology}_rewardfunc{rewardfunc}"
        

    model = SAC(
        "MlpPolicy",
        envs,
        learning_rate=3e-4,  # 学习率
        buffer_size=2_000_000,  # 经验回放缓冲区（加大）
        learning_starts=10_000,  # 多少步后开始训练
        batch_size=1024,  # 增大 batch size，提高 GPU 利用率
        tau=0.005,  # 软更新参数
        gamma=0.99,  # 折扣因子
        train_freq=8,  # 提高数据采样频率
        gradient_steps=4,  # 每次更新 2 个 step，提高利用率
        policy_kwargs=dict(net_arch=[512, 512]),  # 更深的神经网络
        verbose=0,
        device= "cuda",  # 自动选择 GPU/CPU
        tensorboard_log=tensorboard_name
    )
    
    if not callback:
        model.learn(total_timesteps=total_timesteps)  # 训练 100 万步
    else:
        callback = DynamicRewardLoggingCallback()
        model.learn(total_timesteps=total_timesteps, callback=callback)
    
    # print(model_path)
    logging.info(f"Successfully trainning_______morphology{morphology}________rewardfunc{rewardfunc}")
    model.save(model_path)
    envs.close()
    return model_path

def Eva_with_qpos_logging(model_path=None, run_steps=1, folder_name=None, video=False, rewardfunc_index=None, morphology_index=None):
    import numpy as np
    import os
    from stable_baselines3 import SAC


    model = SAC.load(model_path)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "GPTHopper.xml")

    env = FitnessWrapper(gym.make("GPTHopperEnv", xml_file=xml_path,
                                   render_mode="rgb_array", width=1280, height=720))

    fitness_scores = []
    rewards = []
    best_fitness = -np.inf
    best_actions = []

    qpos_log = []  # ⬅️ 用于存储 qpos

    for run in range(run_steps):
        print(f"Run {run}")
        obs, info = env.reset(seed=42)
        actions = []
        total_reward = 0

        for step in range(1000):
            action, _states = model.predict(obs)
            actions.append(action)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward



            if done or truncated:
                fitness = info['fitness']
                fitness_scores.append(fitness)
                rewards.append(total_reward)

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_actions = actions.copy()
                    best_reward = total_reward

                break
    env.close()
    # ✅ 保存 qpos.txt

    
    obs, _ = env.reset(seed=42)  # 保证复现性
    for action in best_actions:
        obs, _, done, _, info = env.step(action)
        qpos = env.unwrapped.data.qpos.copy()
        qpos_log.append(qpos.tolist())
        if done:
            print("fitness",info['fitness'])
            break
    qpos_path = os.path.join(current_dir, "qpos.txt")
    np.savetxt(qpos_path, qpos_log, fmt="%.5f", delimiter=", ")
    print(f"Saved qpos log to {qpos_path}")

    avg_fitness = np.mean(fitness_scores)
    avg_reward = np.mean(rewards)
    print(f"Average Fitness: {avg_fitness:.4f}, Average Reward: {avg_reward:.4f}")

    return avg_fitness, avg_reward


def Eva(model_path=None, run_steps=100, folder_name=None, video = False, rewardfunc_index=None, morphology_index=None):

    model = SAC.load(model_path)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir,"GPTHopper.xml")
    env = FitnessWrapper(gym.make("GPTHopperEnv", xml_file = xml_path, render_mode="rgb_array", width=1280, height=720))
    fitness_scores = []
    rewards = []
    best_fitness = -np.inf
    best_actions = []  # 记录动作序列

    for run in range(run_steps):
        print(run)
        obs, info = env.reset(seed=42)
        actions = []  # 记录当前运行的动作序列

        total_reward = 0
        for step in range(1000):
            action, _states = model.predict(obs)
            actions.append(action)  # 记录动作
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

            if done or truncated:
                # print(f"Step {step}: done={done}, truncated={truncated}, reward={reward}")
            
                fitness = info['fitness']
                fitness_scores.append(fitness)
                rewards.append(total_reward)

                # logging.info(f"Run {run + 1}: Fitness = {fitness} total reward = {total_reward}")

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_actions = actions.copy()  # 保存最高 fitness 的动作序列
                    best_reward = total_reward

                break

    env.close()

    avg_fitness = np.mean(fitness_scores)
    avg_reward = np.mean(rewards)
    var_fitness = np.std(fitness_scores)
    logging.info(f"\nAverage Fitness: {avg_fitness:.4f}  std_fitness {var_fitness} Avg reward {avg_reward:.4f}")


    # 视频录制
    if video:
        video_env = FitnessWrapper(RecordVideo(gym.make("GPTHopperEnv", xml_file = xml_path,  render_mode="rgb_array", width=1280, height=720), video_folder=f"{folder_name}", name_prefix=f"reward{rewardfunc_index}morphology{morphology_index}GPTHopperEnv"))

        obs, _ = video_env.reset(seed=42)  # 保证复现性
        for action in best_actions:
            obs, _, done, _, _ = video_env.step(action)
            if done:
                break

        video_env.close()
        print(f"Video saved")
        
    return avg_fitness, avg_reward


def compute_ant_volume(params):
    """
    params: 
      0: torso半径
      1,2: 第1段胶囊体 x,y
      3,4: 第2段胶囊体 x,y
      5,6: 第3段胶囊体 x,y
      7,8,9: 3 段胶囊体的半径
    """
    # 1) torso 的球体体积
    print(f"params: {params}")

    torso_r =  params[0]

    vol_torso = (4.0/3.0) * math.pi * torso_r**3

    # 2) 计算单条腿 3 段胶囊体
    def capsule_volume(length, radius):
        return math.pi * (radius**2) * length + (4.0/3.0)*math.pi*(radius**3)

    # 第 1 段
    L1 = math.sqrt(params[1]**2 + params[2]**2)
    R1 = params[7]
    v1 = capsule_volume(L1, R1)

    # 第 2 段
    L2 = math.sqrt(params[3]**2 + params[4]**2)
    R2 = params[8]
    v2 = capsule_volume(L2, R2)

    # 第 3 段
    L3 = math.sqrt(params[5]**2 + params[6]**2)
    R3 = params[9]
    v3 = capsule_volume(L3, R3)

    # 单条腿体积
    vol_one_leg = v1 + v2 + v3
    # 4 条腿
    vol_legs = 4 * vol_one_leg

    # 3) 总体积
    vol_total = vol_torso + vol_legs
    return vol_total







def calculate_capsule_volume(radius, height):
    """
    Calculate the volume of a capsule given the radius and height.
    
    Parameters:
    - radius (float): The radius of the capsule.
    - height (float): The height of the capsule.
    
    Returns:
    - volume (float): The volume of the capsule.
    """
    return math.pi * radius**2 * (height + (4 * radius / 3))

def compute_walker_volume(params):
    """
    Compute the total volume of the walker robot model based on the given params.
    
    Parameters:
    - params (list of floats): The input parameters that represent sizes and positions.
    
    Returns:
    - total_volume (float): The total volume consumed by the walker.
    """
    # Assign sizes (radii) for torso, thigh, leg, and foot
    sizes = [params[7], params[8], params[9]]  # These represent sizes {8}, {9}, {10}, etc.

    # Updated heights based on the positional differences (fromto)
    heights_updated = [
        abs(params[1] - params[2]),  # Thigh: |2 - 3|
        abs(params[2] - params[3]),  # Leg: |3 - 4|
        abs(params[4] - params[5])   # Foot: |5 - 6|
    ]

    # Calculate volumes with the updated heights
    volumes_updated = []
    torso_volume = calculate_capsule_volume(params[6], abs(params[0] - params[1]))

    for size, height in zip(sizes, heights_updated):
        volume = calculate_capsule_volume(size, height)
        volumes_updated.append(volume)


    # Return the sum of all volumes
    other_volume = sum(volumes_updated)
    total_volume = torso_volume + 2*other_volume
    return total_volume


def compute_hopper_volume(params):
    """
    Compute the total volume of the hopper robot model based on the given params.
    
    Parameters:
    - params (list of floats): The input parameters that represent sizes and positions.
    
    Returns:
    - total_volume (float): The total volume consumed by the hopper.
    """
    # Assign sizes (radii) for torso, thigh, leg, and foot
    sizes = [params[6], params[7], params[8], params[9]]  # These represent sizes {8}, {9}, {10}, etc.

    # Updated heights based on the positional differences (fromto)
    heights_updated = [
        abs(params[0] - params[1]),  # Torso: |1 - 2|
        abs(params[1] - params[2]),  # Thigh: |2 - 3|
        abs(params[2] - params[3]),  # Leg: |3 - 4|
        abs(params[4] - params[5])   # Foot: |5 - 6|
    ]

    # Calculate volumes with the updated heights
    volumes_updated = []
    for size, height in zip(sizes, heights_updated):
        volume = calculate_capsule_volume(size, height)
        volumes_updated.append(volume)

    # Return the sum of all volumes
    total_volume = sum(volumes_updated)
    return total_volume


class FitnessWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.x_start = None  # 记录初始位置

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.x_start = self.env.unwrapped.data.qpos[0]  # 使用 unwrapped 获取原始环境
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 计算位移
        x_position = self.env.unwrapped.data.qpos[0]  # 访问原始环境的 qpos
        total_distance = x_position - self.x_start  # 计算从起点到当前的位移

        # 计算 fitness
        fitness = total_distance  
        info["fitness"] = fitness  # 把 fitness 存入 info

        return obs, reward, terminated, truncated, info
    
    
