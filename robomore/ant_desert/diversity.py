from openai import OpenAI
from prompts import *
import json
import numpy as np
from design import * 
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull
import seaborn as sns
from utils import *


from openai import OpenAI
from prompts import *
import json
import numpy as np
from design import * 
from utils import *

class DGA:
    def __init__(self):
        api_key = "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"
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-3.5-turbo"
        # self.model = "gpt-4-turbo"

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
            model=self.model, messages=messages, n=1
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
            n=1
        )
        

        initial_parameter = json.loads(response.choices[0].message.content)
        parameter_list.append(initial_parameter['parameters'])
        material_list.append(compute_ant_volume(initial_parameter['parameters']))
        messages.append({"role": "assistant", "content": json.dumps(initial_parameter)})

        print("generate initial_parameter", initial_parameter['parameters'])

        xml_file = ant_design(initial_parameter['parameters'])  

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
            parameter_list.append(diverse_parameter['parameters'])
            messages.append({"role": "assistant", "content": json.dumps(diverse_parameter)})

            xml_file = ant_design(diverse_parameter)  
            filename = f"GPTAnt_{i}.xml"
            file_path = os.path.join(folder_name, "assets", filename)
            with open(file_path, "w") as fp:
                fp.write(xml_file)

            xml_files.append(xml_file)

        return xml_files, material_list, parameter_list


def analyze_morphology_diversity(parameters):
    # 转换参数为数值矩阵
    param_matrix = np.array(parameters)
    
    # 归一化数据，保证不同参数尺度一致
    scaler = StandardScaler()
    param_matrix_scaled = scaler.fit_transform(param_matrix)
    
    # PCA降维到2D
    pca = PCA(n_components=2)
    reduced_params = pca.fit_transform(param_matrix_scaled)
    
    # 计算解释方差
    explained_variance = pca.explained_variance_ratio_
    print(f"Explained Variance Ratio: {explained_variance}")
    print(f"Cumulative Explained Variance: {np.sum(explained_variance)}")
    
    # 使用 seaborn 绘制更美观的散点图
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    sns.set_palette("mako")  # 使用更高级的配色方案
    
    sns.scatterplot(x=reduced_params[:, 0], y=reduced_params[:, 1], hue=None, edgecolor='k', alpha=0.8, s=100)
    
    plt.xlabel("Principal Component 1", fontsize=12, fontweight='bold')
    plt.ylabel("Principal Component 2", fontsize=12, fontweight='bold')
    plt.title("PCA Analysis of Morphology Diversity", fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.show()
    
    return reduced_params  # 返回降维后的数据以便进一步分析



folder_name = setup_logging(div_flag=True)


designer = DGA()

morphology_list, material_list, parameter_list = designer.generate_morphology_div(morphology_nums=5, folder_name = folder_name)
print(parameter_list)
analyze_morphology_diversity(parameter_list)



