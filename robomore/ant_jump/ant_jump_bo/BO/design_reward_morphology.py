def generate_rewardfunc(self, rewardfunc_nums, folder_name):

    messages = [
        {"role": "system", "content": "You are a reinforcement learning reward function designer"},
        {"role": "user", "content": rewardfunc_prompts + zeroshot_rewardfunc_format}
    ]

    responses = self.client.chat.completions.create(
        model=self.model, messages=messages, n=rewardfunc_nums
    )
    files = []
    for i, choice in enumerate(responses.choices):
        reward_code = self.extract_code(choice.message.content)
        if reward_code:
            full_code = self.indent_code(reward_code) + "\n"
            file_name =  f"GPTCheetah_{i}.py"
            file_path = os.path.join(folder_name, "env", file_name)
            with open(file_path, "w") as fp:
                fp.write(full_code)

            with open(file_path, "w") as fp:
                fp.write(full_code)
            files.append(file_path)
            print(f"Saved: {file_path}")
    return files

def generate_rewardfunc_div(self, rewardfunc_nums, folder_name):

    messages = [
        {"role": "system", "content": "You are a reinforcement learning reward function designer"},
        {"role": "user", "content": rewardfunc_prompts + zeroshot_rewardfunc_format}
    ]

    # 生成初始 Reward Function
    response = self.client.chat.completions.create(
        model=self.model, messages=messages, n=1, timeout=10
    )

    rewardfunc_files = []

    initial_code = self.extract_code(response.choices[0].message.content)
    if initial_code:
        reward_code = "import numpy as np\n" + self.indent_code(initial_code) + "\n"

        file_path = os.path.join(folder_name, "env", "GPTrewardfunc_0.py")
        with open(file_path, "w") as fp:
            fp.write(reward_code)
        rewardfunc_files.append(file_path)
        print(f"initial Saved: {file_path}")
    messages.append({"role": "assistant", "content": initial_code})

    # 生成不同的多样化 Reward Functions
    for i in range(1, rewardfunc_nums):
        diverse_messages = messages + [
            {"role": "user", "content": rewardfunc_div_prompts + zeroshot_rewardfunc_format}
        ]
        # print(diverse_messages)
        response = self.client.chat.completions.create(
            model=self.model, messages=diverse_messages, n=1
        )
        diverse_code = self.extract_code(response.choices[0].message.content)
        messages.append({"role": "assistant", "content": diverse_code})

        if diverse_code:
            reward_code =  "import numpy as np\n" + self.indent_code(diverse_code) + "\n"
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

    parameter_list = [json.loads(choice.message.content).get('parameters', []) for choice in responses.choices]
    material_list = [compute_cheetah_volume(parameter) for parameter in parameter_list]

    xml_files = []
    for i, parameter in enumerate(parameter_list):
        if not isinstance(parameter, list):
            print(f"Skipping invalid parameter {i}: {parameter}")
            continue

        xml_file = cheetah_design(parameter)  
        filename = f"GPTCheetah_{i}.xml"
        file_path = os.path.join(folder_name, "assets", filename)
        xml_files.append(file_path)
        with open(file_path, "w") as fp:
            fp.write(xml_file)
        print(f"Successfully saved {filename}")
        
    return xml_files, material_list, parameter_list

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
    material_list.append(compute_cheetah_volume(initial_parameter['parameters']))
    messages.append({"role": "assistant", "content": json.dumps(initial_parameter)})

    logging.info(f"generate initial_parameter{initial_parameter['parameters']}" )

    xml_file = cheetah_design(initial_parameter['parameters'])  

    filename = f"GPTCheetah_0.xml"
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
        material_list.append(compute_cheetah_volume(diverse_parameter['parameters'])) 
        parameter_list.append(diverse_parameter['parameters'])
        messages.append({"role": "assistant", "content": json.dumps(diverse_parameter)})
        logging.info(f"generate diverse_parameter{ diverse_parameter['parameters']}")
        xml_file = cheetah_design(diverse_parameter['parameters'])  
        filename = f"GPTCheetah_{i}.xml"
        file_path = os.path.join(folder_name, "assets", filename)
        with open(file_path, "w") as fp:
            fp.write(xml_file)
        xml_files.append(file_path)

    return xml_files, material_list, parameter_list