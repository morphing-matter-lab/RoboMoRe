# xvfb-run -s "-screen 0 1400x900x24" python test.py 运行这一行就可以得到最终视频了



import time
from design import *
import importlib
import shutil
from utils import *
from openai import OpenAI
from prompts import *
import json
import numpy as np
from gymnasium.envs.robodesign.GPTAnt import GPTAntEnv

folder_name = "results/diverse_rewardfunc"
log_file = os.path.join(folder_name, "parameters.log")
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(message)s")

# folder_name = setup_logging(div_flag=True)

best_fitness = float('-inf')  
best_morphology = None  
best_rewardfunc = None  
best_reward = None
best_material = None
best_efficiency = None

morphology_nums = 2
rewardfunc_nums = 8

fitness_matrix = np.array([[None for _ in range(morphology_nums)] for _ in range(rewardfunc_nums)])
efficiency_matrix = np.array([[None for _ in range(morphology_nums)] for _ in range(rewardfunc_nums)])
fitness_list = []

morphology_list = [f'results/diverse_rewardfunc/assets/GPTAnt_{i}.xml' for i in range(0,2) ]
rewardfunc_list = [f'results/diverse_rewardfunc/env/GPTrewardfunc_{i}.py' for i in range(0,8)]
parameter_list =[ [0.1, 0.35, 0.25, 0.08, 0.12, 0.2, 0.15, 0.015, 0.02, 0.018],
                  [0.25, 0.2, 0.2, 0.2, 0.2, 0.4, 0.4,     0.08, 0.08, 0.08]]


material_list = [compute_ant_volume(parameter) for parameter in parameter_list]


for i, rewardfunc in enumerate(rewardfunc_list):
    for j, morphology in enumerate(morphology_list):
        if j in [0]:
            continue
        print(i, rewardfunc)
        print(j, morphology)
        shutil.copy(morphology, "GPTAnt.xml")
        shutil.copy(rewardfunc, "GPTrewardfunc.py")         

        import GPTrewardfunc
        importlib.reload(GPTrewardfunc)  # 重新加载模块
        from GPTrewardfunc import _get_rew
        GPTAntEnv._get_rew = _get_rew

        env_name = "GPTAntEnv"
        # model_path = Train(j,  i, folder_name, total_timesteps=5e5)
        model_path = f"results/diverse_rewardfunc/coarse/SAC_morphology{j}_rewardfunc{i}_500000.0steps"
        fitness, reward = Eva(model_path=model_path, run_steps=100, folder_name=folder_name, video=True, rewardfunc_index = i, morphology_index = j)
        material = material_list[j]
        efficiency = fitness/material
        fitness_matrix[i][j] = fitness
        efficiency_matrix[i][j] = efficiency
        
        logging.info("___________________finish coarse optimization_____________________")
        logging.info(f"morphology: {j}, rewardfunc: {i}, material cost: {material} reward: {reward} fitness: {fitness} efficiency: {efficiency}")

        if fitness > best_fitness:
            best_fitness = fitness
            best_morphology = morphology
            best_efficiency = efficiency
            best_rewardfunc = rewardfunc
            best_material = material