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
from GPTAnt import GPTAntEnv
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # 确保 Python 能找到当前目录
from stable_baselines3.common.callbacks import BaseCallback




def Eva_with_qpos_logging(model_path=None, run_steps=1, folder_name=None, video=False, rewardfunc_index=None, morphology_index=None):
    import numpy as np
    import os
    from stable_baselines3 import SAC


    model = SAC.load(model_path)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "GPTAnt.xml")

    env = FitnessWrapper(gym.make("GPTAntEnv", xml_file=xml_path, healthy_z_range = (0, 10),
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

# def setup_logging(div_flag):
#     # 生成当前时间作为主文件夹名称
#     if div_flag:
#         folder_name = "div"+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     else:
#         folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     os.makedirs(folder_name, exist_ok=True)

#     # 在 `folder_name` 目录下创建 `assets/` 子文件夹
#     assets_folder = os.path.join(folder_name, "assets")
#     os.makedirs(assets_folder, exist_ok=True)

#     env_folder = os.path.join(folder_name, "env")
#     os.makedirs(env_folder, exist_ok=True)

#     # 创建日志文件路径
#     log_file = os.path.join(folder_name, "parameters.log")

#     # 配置 logging
#     logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(message)s")
    
#     return folder_name  # 返回两个路径


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
        # env = gym.make(env_name, xml_file=r"/home/mml/Ant/GPTAnt.xml")  # 不能加 render_mode，否则 SubprocVecEnv 可能报错
        current_dir = os.path.dirname(os.path.abspath(__file__))
# 构造相对路径
        xml_path = os.path.join(current_dir,"GPTAnt.xml")
        # env = gym.make("Hopper-v5")  # 不能加 render_mode，否则 SubprocVecEnv 可能报错
        env = gym.make("GPTAntEnv", xml_file = xml_path, healthy_z_range = (0, 10))
        return env
    return _init

def Eva_with_qpos_logging2(model_path=None, run_steps=5, folder_name=None, video=False, rewardfunc_index=None, morphology_index=None):
    import numpy as np
    import os
    import csv
    import pandas as pd

    model = SAC.load(model_path)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "GPTAnt.xml")

    env = FitnessWrapper(gym.make("GPTAntEnv", xml_file=xml_path, healthy_z_range=(-10, 10),
                                   render_mode="rgb_array", width=1280, height=720))

    all_qpos0 = []

    for run in range(run_steps):
        print(f"Run {run}")
        obs, info = env.reset(seed=42 + run)
        qpos0_log = []

        for step in range(1000):
            action, _states = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            qpos = env.unwrapped.data.qpos.copy()
            qpos0_log.append(qpos[0])  # 记录 qpos[0]

            if done or truncated:
                break

        all_qpos0.append(qpos0_log)

    env.close()

    # 补齐为等长
    max_len = max(len(seq) for seq in all_qpos0)
    for seq in all_qpos0:
        while len(seq) < max_len:
            seq.append(np.nan)

    df = pd.DataFrame(np.column_stack(all_qpos0), columns=[f"run_{i}" for i in range(run_steps)])
    df['mean'] = df.mean(axis=1)
    df['std'] = df.std(axis=1)
    df['lower'] = df['mean'] - df['std']
    df['upper'] = df['mean'] + df['std']
    df['timestep'] = df.index

    # 重排列顺序为 Origin 格式
    cols = ['timestep', 'mean', 'lower', 'upper'] + [f"run_{i}" for i in range(run_steps)]
    df = df[cols]

    csv_path = os.path.join(current_dir, "qpos0_origin_plot.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved Origin plot CSV to: {csv_path}")
    
    return df

def Train(morphology, rewardfunc, folder_name, total_timesteps = 5e5, stage = 'coarse', callback=False, iter=0):
    num_envs = 16  # 并行环境数量
    envs = SubprocVecEnv([make_env() for _ in range(num_envs)])
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


def Eva(model_path=None, run_steps=100, folder_name=None, video = False, rewardfunc_index=None, morphology_index=None):

    model = SAC.load(model_path)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir,"GPTAnt.xml")
    env = FitnessWrapper(gym.make("GPTAntEnv", xml_file = xml_path, healthy_z_range = (0.05, 10),
render_mode="rgb_array", width=1280, height=720))
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
        video_env = FitnessWrapper(RecordVideo(gym.make("GPTAntEnv", xml_file = xml_path, healthy_z_range = (0.05, 10), render_mode="rgb_array", width=1280, height=720), video_folder=f"{folder_name}", name_prefix=f"reward{rewardfunc_index}morphology{morphology_index}GPTAnt"))

        obs, _ = video_env.reset(seed=42)  # 保证复现性
        for action in best_actions:
            obs, _, done, _, _ = video_env.step(action)
            if done:
                break

        video_env.close()
        print(f"Video saved")
        
    return avg_fitness, avg_reward



def Eva_fitness_record(model_path=None, run_steps=100, folder_name=None,
                      rewardfunc_index=None, morphology_index=None):

    model = SAC.load(model_path)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(current_dir, "GPTAnt.xml")
    env = FitnessWrapper(gym.make("GPTAntEnv", xml_file=xml_path, healthy_z_range=(0.05, 2.0),
                                  render_mode="rgb_array", width=1280, height=720))

    fitness_time_series_all = []
    fitness_scores = []
    rewards = []
    best_fitness = -np.inf
    best_actions = []

    for run in range(run_steps):
        print(f"Run {run+1}/{run_steps}")
        obs, info = env.reset(seed=42)
        actions = []
        fitness_over_time = []
        total_reward = 0

        for step in range(1000):
            action, _states = model.predict(obs)
            actions.append(action)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

            fitness = info.get("fitness", 0)
            fitness_over_time.append((step, fitness))

            if done or truncated:
                fitness_scores.append(fitness)
                rewards.append(total_reward)

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_actions = actions.copy()
                    best_reward = total_reward
                break

        fitness_time_series_all.append(fitness_over_time)

    env.close()

    avg_fitness = np.mean(fitness_scores)
    avg_reward = np.mean(rewards)
    var_fitness = np.std(fitness_scores)

    logging.info(f"\nAverage Fitness: {avg_fitness:.4f}  std_fitness {var_fitness:.4f} Avg reward {avg_reward:.4f}")
    csv_path=f"fitness_{morphology_index}_{morphology_index}.csv"
    # ✅ 导出 CSV：加上仿真时间 time_sec
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["run_index", "step", "time_sec", "fitness"])
        for run_index, series in enumerate(fitness_time_series_all):
            for step, fitness in series:
                time_sec = step * 0.01
                writer.writerow([run_index, step, time_sec, fitness])
        print(f"Saved fitness time series with time info to {csv_path}")

    return avg_fitness, avg_reward, fitness_time_series_all




def TrainEva(folder_name):
    begin = time.time()
    env_name = 'Ant-v5'

    total_timesteps = 2e5

    envs = [lambda: make_env(env_name) for _ in range(16)]

    # 根据参数决定使用哪种向量化方式

    env = DummyVecEnv(envs)

    # envs = SubprocVecEnv([lambda: make_env(env_name) for _ in range(32)]) 
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device="auto",        # 尝试使用 GPU (如果有的话)
        n_steps=2048,         # 可以根据需求调整
        batch_size=64,        # 可以尝试 64, 128, 等看是否加速
        learning_rate=3e-4,   # 可以根据需求调整
        tensorboard_log=folder_name+"./ppo_logs/"
    )

    model.learn(total_timesteps=total_timesteps)
    print("time cost",time.time()-begin)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    env.close()

    # 下面进行fitness评估
    env = make_env(env_name)
    fitness_list = []
    best_fitness = float('-inf')  # 初始化最高 fitness
    steps = 1000

    for run in range(10):
        obs, info = env.reset(seed=42)
        for _ in range(steps):
            action, _states = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action) 
            if done:
                fitness = info['fitness']
                print(f"Run {run + 1}: Fitness = {fitness}")
                fitness_list.append(fitness)
                if fitness > best_fitness:
                    best_fitness = fitness
                break
    mean_fitness = np.mean(fitness_list)
    env.close()
    logging.info(f"Mean fitness: {mean_fitness} , Best Fintess {best_fitness}")

    return mean_reward, mean_fitness


    # model_path = os.path.join(folder_name, f"human_ppo_{env_name}_{total_timesteps}steps.zip")




def retrain_and_save_model(best_parameter, folder_name):
    ant_design(best_parameter) 
    env_name = "Ant-v5"
    total_timesteps = 1e6  # 重新训练更长时间，例如 20 万步
    envs = [lambda: make_env(env_name) for _ in range(16)]
    env = DummyVecEnv(envs)

    # envs = SubprocVecEnv([lambda: make_env(env_name) for _ in range(32)]) 

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device="cpu",        # 尝试使用 GPU (如果有的话)
        n_steps=2048,         # 可以根据需求调整
        batch_size=64,        # 可以尝试 64, 128, 等看是否加速
        learning_rate=3e-4,   # 可以根据需求调整
        tensorboard_log=folder_name+"./ppo_logs/"
    )

    logging.info(f"Retraining with best parameter: {best_parameter}")
    model.learn(total_timesteps=total_timesteps)
    model_path = os.path.join(folder_name, f"best_ppo_{env_name}_{total_timesteps}steps.zip")
    model.save(model_path)


# 下面进行评估
    env = make_env(env_name)
    best_fitness = float('-inf')  # 初始化最高 fitness
    fitness_list = []
    steps = 1000
    for run in range(10):
        obs, info = env.reset(seed=42)
        actions = []  # 记录当前运行的动作序列
        for _ in range(steps):
            action, _states = model.predict(obs)
            actions.append(action)  # 记录动作
            obs, reward, done, truncated, info = env.step(action)
            fitness = info['fitness']
            if done or truncated:
                fitness = info['fitness']
                print(f"Run {run + 1}: Fitness = {fitness}")
                if fitness > best_fitness:
                    best_fitness = fitness
                fitness_list.append(fitness)
                break
    mean_fitness = np.mean(fitness_list)    
    logging.info(f"best parameters: Mean fitness: {mean_fitness} , Best Fintess {best_fitness}")
    env.close()

    # 保存视频
    load_and_visualize_model(model_path, env_name, folder_name)



def load_and_visualize_model(model_path, env_name, folder_name, steps=1000):

    model = PPO.load(model_path)

    env = make_env(env_name)

    # 根据参数决定使用哪种向量化方式


    best_fitness = float('-inf')  # 初始化最高 fitness
    best_actions = None  # 记录最佳动作序列

    for run in range(10):
        obs, info = env.reset(seed=42)
        actions = []  # 记录当前运行的动作序列
        total_fitness = 0
        for _ in range(steps):
            action, _states = model.predict(obs)
            actions.append(action)  # 记录动作
            obs, reward, done, truncated, info = env.step(action)
            fitness = info['fitness']
            total_fitness += fitness

            if done or truncated:
                fitness = info.get('fitness', total_fitness) 
                print(f"Run {run + 1}: Fitness = {fitness}")
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_actions = actions.copy()  # 保存最高 fitness 的动作序列
                break

    env.close()

    video_env = RecordVideo(
        gym.make(env_name, xml_file=os.path.join(os.path.dirname(__file__), "assets\GPTAnt.xml"), healthy_z_range= (0.2, 1.8), width=1280, height=720),
        video_folder=folder_name,
        name_prefix="best_fitness_run"
    )

    obs, _ = video_env.reset(seed=42)  # 保证复现性
    for action in best_actions:
        obs, _, done, _, _ = video_env.step(action)
        if done:
            break

    video_env.close()



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
