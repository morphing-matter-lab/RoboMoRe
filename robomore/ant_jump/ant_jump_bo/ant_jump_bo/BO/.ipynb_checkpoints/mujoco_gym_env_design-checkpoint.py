import os
import warnings
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Insert at beginning of path to prioritize local version
sys.path.insert(0, project_root)

import gymnasium as gym
import numpy as np
import torch

# print("Gymnasium path:", gym.__file__)
# print("Gym path:", gym.__path__)
# print("Python path:", sys.path)

from design import *
from utils import Train, Eva


class MujocoGymEnv():
    # Avaliable environments:
    # https://www.gymlibrary.ml/environments/mujoco/
    def __init__(self, env_name, num_rollouts, minimize=True):
        assert num_rollouts > 0
        self.minimum = minimize
        self.num_rollouts = num_rollouts
        self.mean, self.std = self.get_mean_std(env_name)
        self.env = gym.vector.make(env_name, num_envs=num_rollouts)
        self.__name__ = env_name
        assert self.env.action_space.shape[0] == num_rollouts and self.env.observation_space.shape[0] == num_rollouts
        self.obs_shape = self.env.observation_space.shape[1]
        self.act_shape = self.env.action_space.shape[1]

        self.weight_matrix = self.build_weight_matrix((self.act_shape, self.obs_shape))
        self.dim = self.obs_shape * self.act_shape

        self.lb = -1 * np.ones(self.dim)
        self.ub = 1 * np.ones(self.dim)

    def get_mean_std(self, env_name):
        env_name = env_name.split('-')[0]
        if env_name in ['Ant', 'HalfCheetah', 'Hopper', 'Walker2d', 'Humanoid', 'Swimmer']:
            file_path = os.path.dirname(__file__) + '/trained_policies/' + env_name + '-v1/lin_policy_plus.npz'
            data = np.load(file_path, allow_pickle=True)['arr_0']
            mean = data[1]
            std = data[2]
            return mean, std
        else:
            warnings.warn('No mean and std for this environment')
            return 0, 1

    def reset(self):
        return self.env.reset(seed=0)
        # return self.env.reset() # noise

    def step(self, action):
        return self.env.step(action)

    def build_weight_matrix(self, shape):
        return np.random.randn(*shape)

    def get_action(self, obs):
        return np.dot(self.weight_matrix, obs.T)

    def update_weight_matrix(self, updated_weight_matrix):
        if updated_weight_matrix.shape != self.weight_matrix.shape:
            updated_weight_matrix = updated_weight_matrix.reshape(
                self.weight_matrix.shape)
        self.weight_matrix = updated_weight_matrix

    def __call__(self, updated_weight_matrix):
        if isinstance(updated_weight_matrix, torch.Tensor):
            updated_weight_matrix = updated_weight_matrix.detach().cpu().numpy()
        
        updated_weight_matrix = np.clip(updated_weight_matrix, self.lb, self.ub)
        assert np.all(updated_weight_matrix <= self.ub) and np.all(
            updated_weight_matrix >= self.lb)
        self.update_weight_matrix(updated_weight_matrix)

        # obs = self.reset()
        obs = self.reset()[0]

        # print(f'obs {obs.shape}')
        done = [False for _ in range(self.num_rollouts)]
        truncated = [False for _ in range(self.num_rollouts)]
        totalReward = [0 for _ in range(self.num_rollouts)]
        # print(f'new one {round(time.time())}')
        while not any(done) and not any(truncated):
            
            obs = (obs - self.mean) / self.std
            action = self.get_action(obs).T
            obs, reward, done,truncated, info = self.step(action)
            # obs, reward, done, info = self.step(action)

            # print(action)
            # print(obs.shape)
            totalReward = [i + j for i, j in zip(totalReward, reward)]
            # print(totalReward)
        # print(totalReward)
        # raise NotImplementedError
        if not self.minimum:
            return np.mean(totalReward)
        else:
            return -1 * np.mean(totalReward)

def test_dual_annealing():
    from scipy.optimize import dual_annealing
    mjEnv = MujocoGymEnv("Swimmer-v2", 3)
    ret = dual_annealing(mjEnv, bounds=[(-1, 1) for _ in range(16)])
    print(ret)

def test_differential_evolution():
    from scipy.optimize import differential_evolution
    mjEnv = MujocoGymEnv("Swimmer-v2", 3)
    ret = differential_evolution(mjEnv, bounds=[(-1, 1) for _ in range(16)])
    print(ret)


if __name__ == "__main__":
    # test_dual_annealing()
    # test_differential_evolution()
    func = MujocoGymEnv('Humanoid-v2', 1, minimize=False)

    x = np.ones(func.dim)
    for i in range(10):
        print(func(x))
    

class MujocoDesignEnv():
    def __init__(self, env_name, num_rollouts, minimize=False):
        self.minimum = minimize
        self.num_rollouts = num_rollouts
        self.__name__ = f"{env_name}_design"
        
        if env_name == 'GPTAntEnv':
            self.dim = 10
            
            self.lb = np.array([
                0.01,    # Torso radius
                0.01,     # Leg segment 1 x
                0.01,     # Leg segment 1 y
                0.01,     # Leg segment 2 x
                0.01,     # Leg segment 2 y
                0.01,     # Foot x
                0.01,     # Foot y
                0.01,    # Leg segment 1 radius
                0.01,    # Leg segment 2 radius
                0.01,    # Foot radius
            ])
            
            self.ub = np.array([
                0.5,     # Torso radius
                0.5,     # Leg segment 1 x
                0.5,     # Leg segment 1 y
                0.5,     # Leg segment 2 x
                0.5,     # Leg segment 2 y
                0.5,     # Foot x
                0.5,     # Foot y
                0.5,    # Leg segment 1 radius
                0.5,    # Leg segment 2 radius
                0.5,    # Foot radius
            ])
            
        else:
            raise NotImplementedError(f"Environment {env_name} not supported")

    def __call__(self, design_params):

        print("calling MujocoDesignEnv for training and evaluation")
        if isinstance(design_params, torch.Tensor):
            design_params = design_params.detach().cpu().numpy()
        
        # Ensure parameters are within bounds
        design_params = np.clip(design_params, self.lb, self.ub)
        init_params = np.array([0.25, 0.2, 0.2, 0.2, 0.2, 0.4, 0.4, 0.08, 0.08, 0.08])
        design_params += init_params
        
        try:
            # Generate XML using ant_design function
            xml_string = ant_design(design_params)
            
            # Save to a temporary file first
            tmp_xml = f"/tmp/ant_design_{os.getpid()}.xml"
            with open(tmp_xml, "w") as f:
                f.write(xml_string)
            
            # Copy to standard name that Train/Eva expect
            import shutil
            shutil.copy(tmp_xml, "GPTAnt.xml")
            
            # Create folder for training results
            folder_name = f"/tmp/train_results_{os.getpid()}"
            os.makedirs(folder_name, exist_ok=True)

            print(f"Running Train() with {design_params}")
            
            # Train with default reward function (index 0)
            model_path = Train(morphology=os.getpid(), 
                             rewardfunc=0, # Refer to default reward function in GPTAntEnv
                             folder_name=folder_name,
                             total_timesteps=5e5)

            print("Train() finished, running Eva()")
            
            # Evaluate the trained agent
            avg_fitness, avg_reward = Eva(model_path=model_path)
            material = compute_ant_volume(design_params)
            efficiency = avg_fitness/material
            
            os.remove(tmp_xml)
            
            # Return negative reward if minimizing, positive if maximizing
            return -efficiency if self.minimum else efficiency
            
        except Exception as e:
            print(f"Error evaluating design: {e}")
            return float('-inf') if not self.minimum else float('inf')
    