import time
from design import *
from utils import *
import importlib
import shutil
from openai import OpenAI
from prompts import *
import json
import numpy as np
from gymnasium.envs.robodesign.GPTAnt import GPTAntEnv



from prompts import *
class DGA:
    def __init__(self):
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"
        # self.model = "gpt-3.5-turbo"

    def extract_code(self, text):
        match = re.search(r'```python\n(.*?)\n```', text, re.DOTALL)
        return match.group(1).strip() if match else None

    def indent_code(self, code):
        return "\n".join("    " + line if line.strip() else line for line in code.split("\n"))

    def generate_rewardfunc(self, rewardfunc_nums, folder_name):
        env_path = os.path.join(os.path.dirname(__file__), "env", "ant_v5.py")
        with open(env_path, "r") as f:
            env_content = f.read().rstrip()

        messages = [
            {"role": "system", "content": "You are a reinforcement learning reward function designer"},
            {"role": "user", "content": rewardfunc_prompts + rewardfunc_format}
        ]

        responses = self.client.chat.completions.create(
            model=self.model, messages=messages, n=rewardfunc_nums
        )
        files = []
        for i, choice in enumerate(responses.choices):
            reward_code = self.extract_code(choice.message.content)
            if reward_code:
                full_code = env_content + "\n\n" + self.indent_code(reward_code) + "\n"
                file_name =  f"GPTAnt_{i}.py"
                file_path = os.path.join(folder_name, "env", file_name)
                with open(file_path, "w") as fp:
                    fp.write(full_code)

                with open(file_path, "w") as fp:
                    fp.write(full_code)
                files.append(file_path)
                print(f"Saved: {file_path}")
        return files
    
    def generate_rewardfunc_div(self, rewardfunc_nums, folder_name):

        # env_path = os.path.join(os.path.dirname(__file__), "env", "ant_v5.py")
        # with open(env_path, "r") as f:
        #     env_content = f.read().rstrip()

        messages = [
            {"role": "system", "content": "You are a reinforcement learning reward function designer"},
            {"role": "user", "content": rewardfunc_prompts + rewardfunc_format}
        ]

        # 生成初始 Reward Function
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, n=1, timeout=10
        )

        rewardfunc_files = []

        initial_code = self.extract_code(response.choices[0].message.content)
        if initial_code:
            reward_code = self.indent_code(initial_code) + "\n"

            file_path = os.path.join(folder_name, "env", "GPTrewardfunc_0.py")
            with open(file_path, "w") as fp:
                fp.write(reward_code)
            rewardfunc_files.append(file_path)
            print(f"initial Saved: {file_path}")

        # 生成不同的多样化 Reward Functions
        for i in range(1, rewardfunc_nums+1):
            diverse_messages = messages + [
                {"role": "user", "content": rewardfunc_div_prompts + rewardfunc_format}
            ]

            response = self.client.chat.completions.create(
                model=self.model, messages=diverse_messages, n=1
            )

            diverse_code = self.extract_code(response.choices[0].message.content)
            if diverse_code:
                reward_code =  "\n\n" + self.indent_code(diverse_code) + "\n"
                file_path = os.path.join(folder_name, "env", f"GPTrewardfunc_{i}.py")
                with open(file_path, "w") as fp:
                    fp.write(reward_code)
                rewardfunc_files.append(file_path)
                print(f"Saved: {file_path}")

        return rewardfunc_files

    def generate_morphology(self, morphology_nums, folder_name):
        messages = [
            {"role": "system", "content": "You are a helpful mujoco robot designer"},
            {"role": "user", "content": morphology_prompts + morphology_format}
        ]
        
        responses = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={'type': 'json_object'},
            n=morphology_nums
        )

        # 解析所有 response 里的参数
        for i, choice in enumerate(responses.choices):
            print(f"Response {i}:")
            print(json.dumps(choice.message.content, indent=4))

        parameters_list = [json.loads(choice.message.content).get('parameters', []) for choice in responses.choices]


        xml_files = []
        for i, parameter in enumerate(parameters_list):
            if not isinstance(parameter, list):
                print(f"Skipping invalid parameter {i}: {parameter}")
                continue

            xml_file = ant_design(parameter)  
            filename = f"GPTAnt_{i}.xml"
            # file_path = os.path.join(os.path.dirname(__file__), folder_name, "assets", filename)
            file_path = os.path.join(folder_name, "assets", filename)
            # print(file_path)

            with open(file_path, "w") as fp:
                fp.write(xml_file)

            print(f"Successfully saved {filename}")
            xml_files.append(file_path)
        return xml_files
    
    def generate_morphology_div(self, morphology_nums, folder_name):

        material_list = []
        xml_files = []
        parameter_list = []
        
        # 生成初始 morphology
        messages = [
            {"role": "system", "content": "You are a helpful mujoco robot designer"},
            {"role": "user", "content": morphology_prompts + morphology_format}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={'type': 'json_object'},
            n=1
        )
        

        initial_parameter = json.loads(response.choices[0].message.content)
        parameter_list.append(initial_parameter['parameters'])
        material_list.append(compute_ant_volume(initial_parameter['parameters']))
        messages.append({"role": "assistant", "content": json.dumps(initial_parameter)})

        logging.info(f"generate initial_parameter{initial_parameter['parameters']}" )

        xml_file = ant_design(initial_parameter['parameters'])  

        filename = f"GPTAnt_0.xml"
        file_path = os.path.join(folder_name, "assets", filename)
        with open(file_path, "w") as fp:
            fp.write(xml_file)

        xml_files.append(file_path)

        # 生成不同的多样化设计
        for i in range(1, morphology_nums):
            diverse_messages = messages + [
                {"role": "user", "content": morphology_div_prompts + morphology_format}
            ]
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=diverse_messages,
                response_format={'type': 'json_object'},
                n=1
            )

            diverse_parameter = json.loads(response.choices[0].message.content)
            material_list.append(compute_ant_volume(diverse_parameter['parameters']))
            parameter_list.append(diverse_parameter['parameters'])
            messages.append({"role": "assistant", "content": json.dumps(diverse_parameter)})
            logging.info(f"generate diverse_parameter{ diverse_parameter['parameters']}")
            xml_file = ant_design(diverse_parameter['parameters'])  
            filename = f"GPTAnt_{i}.xml"
            file_path = os.path.join(folder_name, "assets", filename)
            with open(file_path, "w") as fp:
                fp.write(xml_file)
            xml_files.append(filename)

        return xml_files, material_list, parameter_list


    def improve_rewardfunc(self, best_rewardfunc, rewardfunc_list, fitness_list, folder_name):
        reward_improve_prompts = prompts.reward_improve_prompts
        for reward_content, fitness in zip(rewardfunc_list, fitness_list):
            reward_improve_prompts = reward_improve_prompts + f"reward function:{reward_content} \n" + f"fintess:{fitness}"
        reward_improve_prompts = reward_improve_prompts + f"best reward function:{best_rewardfunc} \n" + f"best fintess:{max(fitness_list)}" 

        messages = [
            {"role": "system", "content": "You are a reinforcement learning reward function designer"},
            {"role": "user", "content": rewardfunc_prompts + rewardfunc_format}
        ]

        response = self.client.chat.completions.create(
            model=self.model, messages=messages
        )

        print(response)
        reward_code = self.extract_code(response.choices[0].message.content)

        if reward_code:
            full_code = "import numpy as np \n" + self.indent_code(reward_code) + "\n"
            file_name =  f"GPTAnt_refine.py"

            file_path = os.path.join(folder_name, "env", file_name)
            with open(file_path, "w") as fp:
                fp.write(full_code)

        return file_path

    def improve_morphology(self, best_parameter, parameter_list, fitness_list, folder_name):
        morphology_improve_prompts = prompts.morphology_improve_prompts
        for parameter_content, fitness in zip(parameter_list, fitness_list):
            morphology_improve_prompts = morphology_improve_prompts + f"parameter:{parameter_content} \n" + f"fintess:{fitness}"
        morphology_improve_prompts = morphology_improve_prompts + f"best parameter:{best_parameter} \n" + f"best fintess:{max(fitness_list)}" 

        messages = [
            {"role": "system", "content": "You are a helpful mujoco robot designer"},
            {"role": "user", "content": morphology_improve_prompts + morphology_format}
        ]
        
        responses = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={'type': 'json_object'},
        )
        print(responses)
        parameter = json.loads(responses.choices[0].message.content).get('parameters', []) 
        print(parameter)
        xml_file = ant_design(parameter)  
        filename = f"GPTAnt_refine.xml"
        file_path = os.path.join(folder_name, "assets", filename)

        with open(file_path, "w") as fp:
            fp.write(xml_file)

        print(f"Successfully saved {filename}")
        return file_path, parameter

    
folder_name = setup_logging(div_flag=True)
# folder_name = "results/gpt4turbo"

# log_file = os.path.join(folder_name, "parameters.log")
# logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(message)s")

best_fitness = float('-inf')  
best_morphology = None  
best_rewardfunc = None  
best_reward = None
best_material = None
best_material_efficiency = None

morphology_nums = 8
rewardfunc_nums = 5
fitness_matrix = [[None for _ in range(morphology_nums)] for _ in range(rewardfunc_nums)]
fitness_list = []
designer = DGA()



# return file list of morphology and reward function: [GPTAnt_{i}.xml] and [GPTAnt_{j}.py]

morphology_list, material_list, parameter_list = designer.generate_morphology_div(morphology_nums, folder_name)


rewardfunc_list = designer.generate_rewardfunc_div(rewardfunc_nums, folder_name)


logging.info(f'folder_name:{folder_name}')
logging.info(f'morphology_nums:{morphology_nums}')
logging.info(f'rewardfunc_nums:{rewardfunc_nums}')
logging.info(f'parameter_list:{parameter_list}')
logging.info(f'morphology_list:{morphology_list}')
logging.info(f'material_list:{material_list}')
logging.info(f'_________________________________enter coarse optimization stage_________________________________')



# morphology_list = [f'results/gpt4o/assets/GPTAnt_{i}.xml' for i in range(0,9) ]
# rewardfunc_list = [f'results/gpt4o/env/GPTrewardfunc_{i}.py' for i in range(0,6)]

for j, rewardfunc in enumerate(rewardfunc_list):
    for i, morphology in enumerate(morphology_list):
        # if i <= 4:
        #     continue
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
        material = material_list[i]
        merterial_efficiency = fitness/material
        fitness_matrix[i][j] = fitness

        logging.info(f"morphology: {i}, rewardfunc: {j}, material cost: {material} reward: {reward} fitness: {fitness} merterial_efficiency: {merterial_efficiency}")

        if fitness > best_fitness:
            best_fitness = fitness
            best_morphology = morphology
            best_material_efficiency = merterial_efficiency
            best_rewardfunc = rewardfunc
            best_material = material
            
logging.info(f"Stage1: Final best morphology: {best_morphology}, best reward function: {best_rewardfunc}, Material cost: {best_material}, Reward: {best_reward}, Fitness: {best_fitness}, best_material_efficiency: {best_material_efficiency}")
logging.info(f'folder_name:{folder_name}')
logging.info(f'coarse_best:{coarse_best}')
logging.info(f'parameter_list:{parameter_list}')
logging.info(f'fitness_matrix:{fitness_matrix}')
logging.info(f'_________________________________enter fine optimization stage_________________________________')



for morphology_index, rewardfunc_index in coarse_best:
    morphology = morphology_list[morphology_index]
    parameter = parameter_list[morphology_index]
    rewardfunc = rewardfunc_list[rewardfunc_index]
    best_fitness = fitness_matrix[morphology_index][rewardfunc_index]

    print("morphology", morphology)
    print("parameter", parameter)
    print("rewardfunc", rewardfunc)
    print("best_fitness", best_fitness)

    while True:
        designer = DGA()
        # 输入最好的morphology & rewardfunc, 之前所有的参数parameter_list，以及对应的fitness_matrix
        improved_rewardfunc = designer.improve_rewardfunc(rewardfunc, rewardfunc_list, fitness_matrix[morphology_index], folder_name)
        shutil.copy(morphology, "GPTAnt.xml")
        shutil.copy(improved_rewardfunc, "GPTrewardfunc.py")
        # shutil.copy(rewardfunc, "GPTrewardfunc.py")
        model_path = Train(morphology_index, rewardfunc_index, folder_name)
        improved_fitness, improved_reward = Eva(model_path)
        improved_material = compute_ant_volume(parameter_list[morphology_index])
        improved_material_efficiency = improved_fitness/improved_material
        
        if improved_fitness>best_fitness:
            best_fitness = improved_fitness
            best_morphology = morphology
            best_parameter = parameter
            best_material_efficiency = improved_material_efficiency
            best_rewardfunc = improved_rewardfunc
            best_material = improved_material
            logging.info(f"reward optimization: material cost: {improved_material}  fitness: {improved_fitness} merterial_efficiency: {improved_material_efficiency}")
        else:
            break
            
            
        improved_morphology, improved_parameter = designer.improve_morphology(parameter, parameter_list, fitness_matrix[:,rewardfunc_index], folder_name)
        shutil.copy(improved_morphology, "GPTAnt.xml")
        shutil.copy(best_rewardfunc, "GPTrewardfunc.py")
        model_path = Train(morphology_index,  rewardfunc_index, folder_name)
        improved_fitness, improved_reward = Eva(model_path)
        improved_material = compute_ant_volume(improved_parameter)
        improved_material_efficiency = improved_fitness/improved_material

        if improved_fitness>best_fitness:
            best_fitness = improved_fitness
            best_morphology = improved_morphology
            best_parameter = improved_parameter
            best_material_efficiency = improved_material_efficiency
            best_rewardfunc = best_rewardfunc
            best_material = improved_material
            logging.info(f"morphology optimization: material cost: {improved_material}  fitness: {improved_fitness} merterial_efficiency: {improved_material_efficiency}")

        else:
            break

            
        rewardfunc = best_rewardfunc
        morphology = best_morphology
        parameter = best_parameter