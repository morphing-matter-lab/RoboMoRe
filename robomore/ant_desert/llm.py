from openai import OpenAI
from prompts import *
import json
import numpy as np
from design import * 
from utils import *

class DGA:
    def __init__(self):
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"

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
    
    def generate_rewardfunc_div(self, rewardfunc_nums):

        env_path = os.path.join(os.path.dirname(__file__), "env", "ant_v5.py")
        with open(env_path, "r") as f:
            env_content = f.read().rstrip()

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
            reward_code = env_content + "\n\n" + self.indent_code(initial_code) + "\n"
            file_path = os.path.join(os.path.dirname(__file__), "env", "GPTrewardfunc_0.py")
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
                file_path = os.path.join(os.path.dirname(__file__), "env", f"GPTrewardfunc_{i}.py")
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
            n=1,
            timeout=10
        )

        initial_parameter = json.loads(response.choices[0].message.content)
        material_list.append(compute_ant_volume(initial_parameter['parameters']))
        messages.append({"role": "assistant", "content": initial_parameter})
        parameter_list.append(initial_parameter['parameters'])

        xml_file = ant_design(initial_parameter)  
        filename = f"GPTAnt_0_div.xml"
        file_path = os.path.join(folder_name, "assets", filename)
        with open(file_path, "w") as fp:
            fp.write(xml_file)

        xml_files.append(xml_file)

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
            messages.append({"role": "assistant", "content": diverse_parameter})
            parameter_list.append(diverse_parameter['parameters'])

            xml_file = ant_design(diverse_parameter)  
            filename = f"GPTAnt_{i}.xml"
            file_path = os.path.join(folder_name, "assets", filename)
            with open(file_path, "w") as fp:
                fp.write(xml_file)

            xml_files.append(xml_file)

        return parameter_list, xml_files, material_list


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


    
    
    # def generate_rewardfunc_div(self, rewardfunc_nums):
    #     env_path = os.path.join(os.path.dirname(__file__), "env", "ant_v5.py")
    #     with open(env_path, "r") as f:
    #         env_content = f.read().rstrip()

    #     messages = [
    #         {"role": "system", "content": "You are a reinforcement learning reward function designer"},
    #         {"role": "user", "content": "Write a random reward function" + rewardfunc_format}
    #     ]

    #     # 生成初始 Reward Function
    #     response = self.client.chat.completions.create(
    #         model=self.model, messages=messages, n=1
    #     )

    #     rewardfunc_files = []

    #     initial_code = self.extract_code(response.choices[0].message.content)
    #     if initial_code:
    #         full_code = env_content + "\n\n" + self.indent_code(initial_code) + "\n"
    #         file_path = os.path.join(os.path.dirname(__file__), "env", "GPTant_0.py")
    #         with open(file_path, "w") as fp:
    #             fp.write(full_code)
    #         rewardfunc_files.append(file_path)
    #         print(f"Saved: {file_path}")

    #     # 生成不同的多样化 Reward Functions
    #     for i in range(1, rewardfunc_nums+1):
    #         diverse_messages = messages + [
    #             {"role": "user", "content": rewardfunc_div_prompts + rewardfunc_format}
    #         ]

    #         response = self.client.chat.completions.create(
    #             model=self.model, messages=diverse_messages, n=1
    #         )

    #         diverse_code = self.extract_code(response.choices[0].message.content)
    #         if diverse_code:
    #             full_code = env_content + "\n\n" + self.indent_code(diverse_code) + "\n"
    #             file_path = os.path.join(os.path.dirname(__file__), "env", f"GPTAnt_div_{i}.py")

    #             with open(file_path, "w") as fp:
    #                 fp.write(full_code)
    #             rewardfunc_files.append(file_path)
    #             print(f"Saved: {file_path}")

    #     return rewardfunc_files