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

    

# Ensure parameters are within bounds
design_params = parameter =  [0.009999999776482582,0.009999999776482582,0.5,0.009999999776482582,0.009999999776482582,0.009999999776482582]
init_parameter = np.array([1, 1, 1, 0.1, 0.1, 0.1])
design_params = init_parameter + design_params

try:
    # Generate XML using ant_design function
    xml_string = swimmer_design(design_params)

    # Save to a temporary file first
    tmp_xml = f"/tmp/swimmer_design_{os.getpid()}.xml"
    with open(tmp_xml, "w") as f:
        f.write(xml_string)

    # Copy to standard name that Train/Eva expect
    import shutil
    shutil.copy(tmp_xml, "GPTSwimmer.xml")

    # Create folder for training results
    folder_name = f"/tmp/train_results_{os.getpid()}"
    os.makedirs(folder_name, exist_ok=True)

    print(f"Running Train() with {design_params}")

    # Train with default reward function (index 0)
    model_path = Train(morphology=os.getpid(), 
                     rewardfunc=0, # Refer to default reward function in GPTAntEnv
                     folder_name=folder_name,
                     total_timesteps=1e6)

    print("Train() finished, running Eva()")

    # Evaluate the trained agent
    avg_fitness, avg_reward = Eva(model_path=model_path)
    material = compute_swimmer_volume(design_params)
    efficiency = avg_fitness/material
    os.remove(tmp_xml)

    print(avg_fitness,efficiency)

except Exception as e:
    print(f"Error evaluating design: {e}")
    return 0
    # return float('-inf') if not self.minimum else float('inf')
    