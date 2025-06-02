import time
from design import *
from llm import *
from utils import *
import importlib
import shutil
from gymnasium.envs.robodesign.GPTAnt import GPTAntEnv
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # 确保 Python 能找到当前目录

if __name__ == "__main__":

    fitness_list = []
    # folder_name = setup_logging(div_flag=True)
    # folder_name = "1design"
    folder_name = "results/gpt4o"
    
    log_file = os.path.join(folder_name, "parameters.log")
    logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(message)s")

    best_fitness = float('-inf')  
    best_morphology = None  
    best_rewardfunc = None  
    best_reward = None
    best_material = None
    best_material_efficiency = None

    morphology_nums = 1
    rewardfunc_nums = 1

    # designer = DGA()
    # morphology_list, material_list = designer.generate_morphology_div(morphology_nums, folder_name)
    # rewardfunc_list = designer.generate_rewardfunc_div(rewardfunc_nums, folder_name)
    
    # return file list of morphology and reward function: [GPTAnt_{i}.xml] and [GPTAnt_{j}.py]
    morphology_list = [f'results/gpt4o/assets/GPTAnt_{i}.xml' for i in range(0,10) ]
    rewardfunc_list = [f'results/gpt4o/env/GPTrewardfunc_{i}.py' for i in range(0,6)]
    # morphology_list = [f'results/gpt4o/assets/GPTAnt_8.xml' ]
    # rewardfunc_list = [f'results/gpt4o/env/GPTrewardfunc_0.py']


# step1. Coarse Optmization
    for i, morphology in enumerate(morphology_list):
        for j, rewardfunc in enumerate(rewardfunc_list):
            if i not in [9]:
                continue
                
            print(i, morphology)
            print(j, rewardfunc)
            shutil.copy(morphology, "GPTAnt.xml")
            shutil.copy(rewardfunc, "GPTrewardfunc.py")


            import GPTrewardfunc
            importlib.reload(GPTrewardfunc)  # 重新加载模块
            from GPTrewardfunc import _get_rew
            GPTAntEnv._get_rew = _get_rew


            env_name = "GPTAntEnv"
            model_path = Train(i,  j, folder_name)

            # model_path = "1design/1design/assets/GPTAnt_0.xml_GPTAntEnv_sac_300000.0steps"
            fitness, reward = Eva(model_path)
            # material = material_list[i]
            material = 1
            merterial_efficiency = fitness/material
            fitness_list.append(fitness)

            logging.info(f"morphology: {i}, rewardfunc: {j}, material cost: {material} reward: {reward} fitness: {fitness} merterial_efficiency: {merterial_efficiency}")

            if fitness > best_fitness:
                best_fitness = fitness
                best_morphology = morphology
                best_material_efficiency = merterial_efficiency
                best_reward = reward
                best_rewardfunc = rewardfunc
                best_material = material

    logging.info(f"Stage1: Final best morphology: {best_morphology}, best reward function: {best_rewardfunc}, Material cost: {best_material}, Reward: {best_reward}, Fitness: {best_fitness}, best_material_efficiency: {best_material_efficiency}")





# step2. Fine Optmization, 筛选出最好的rewardfunc和morphology后
    # 1. 让其观察之前的reward function，并提出更好的reward function

    # while():
    #     improved_rewardfunc = designer.improve_rewardfunc()
    #     train(best_morphology, improved_rewardfunc)


    #     material = cal_material(best_morphology)
    #     merterial_efficiency = fitness/material
    #     fitness_list.append(fitness)
    #     if fitness>best_fitness:
    #         best_fitness = fitness
    #         best_morphology = morphology
    #         best_material_efficiency = merterial_efficiency
    #         best_reward = reward
    #         best_rewardfunc = rewardfunc
    #         best_material = material
    #     else:
    #         break


    #     improved_morphology = designer.improve_morphology()
    #     train(best_morphology, improved_morphology)

    #     material = cal_material(improved_morphology)
    #     merterial_efficiency = fitness/material
    #     fitness_list.append(fitness)
    #     if fitness>best_fitness:
    #         best_fitness = fitness
    #         best_morphology = morphology
    #         best_material_efficiency = merterial_efficiency
    #         best_reward = reward
    #         best_rewardfunc = rewardfunc
    #         best_material = material
    #     else:
    #         break
    # logging.info(f"Stage2: Final best morphology: {best_morphology}, best reward function: {best_rewardfunc}, Material cost: {best_material}, Reward: {best_reward}, Fitness: {best_fitness}, best_material_efficiency: {best_material_efficiency}")





    # retrain_and_save_model(best_morphology, folder_name)







