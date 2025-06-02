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
from design import cheetah_design

# print("Gymnasium path:", gym.__file__)
# print("Gym path:", gym.__path__)
# print("Python path:", sys.path)

from design import *
from utils import Train, Eva
import shutil
import importlib
from gymnasium.envs.robodesign.GPTCheetah import GPTCheetahEnv


import GPTrewardfunc
importlib.reload(GPTrewardfunc)  # 重新加载模块
from GPTrewardfunc import _get_rew
GPTCheetahEnv._get_rew = _get_rew



# Ensure parameters are within bounds
design_params =np.array([0.5,0.009999999776482582,0.009999999776482582,0.009999999776482582,0.5,0.4671628177165985,0.018577031791210175,0.009999999776482582,0.1319998949766159,0.30846139788627625,0.009999999776482582,0.009999999776482582,0.009999999776482582,0.5,0.5,0.5,0.009999999776482582,0.009999999776482582,0.009999999776482582,0.009999999776482582,0.009999999776482582,0.009999999776482582,0.009999999776482582,0.009999999776482582])
offsets = np.array([-0.6, 0.6, 0.8, 0.2, 0.17744, -0.22938, -0.26892,-0.13297,0.05015, -0.18119, -0.13217, -0.23084, 0.11970, -0.17497, 0.07905, -0.11555, 0.04600, 0.04600, 0.04600, 0.04600, 0.04600, 0.04600, 0.04600, 0.04600]
)

design_params += offsets

# Generate XML using Cheetah_design function
xml_string = cheetah_design(design_params)

# Save to a temporary file first
tmp_xml = f"cheetah_design.xml"
with open(tmp_xml, "w") as f:
    f.write(xml_string)

# Copy to standard name that Train/Eva expect

shutil.copy(tmp_xml, "GPTCheetah.xml")

# Create folder for training results
folder_name = f"/tmp/train_results_{os.getpid()}"
os.makedirs(folder_name, exist_ok=True)

print(f"Running Train() with {design_params}")

morphology_index=222
rewardfunc_index=222

model_path = Train(morphology_index, rewardfunc_index, folder_name, stage='fine', total_timesteps=1e6)

fitness, _ = Eva(model_path)
material = compute_cheetah_volume(parameter)
efficiency = fitness / material

print(f"fitness:{fitness}")
print(f"efficiency:{efficiency}")

with open('eff_fit.txt', 'w') as f:

        f.write(f"fitness:{fitness}")
        f.write(f"efficiency:{efficiency}")

        f.write("\n")