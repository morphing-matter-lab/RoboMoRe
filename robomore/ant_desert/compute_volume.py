import math

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
    torso_r = params[0]
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

# 演示调用
if __name__ == "__main__":
    # 随便举个参数例子







# 极致跃迁——高速弹跳型四足机器人
    json_param = {
    "parameters": [
        0.05, 
        0.25, 
        0.4, 
        0.35, 
        0.5, 
        0.4, 
        0.6, 
        0.02, 
        0.04, 
        0.05
    ]
    }


# 稳定至上——低重心、强支撑型步行机器人
    # json_param = {
    # "parameters": [
    #     0.3, 
    #     0.1, 
    #     0.15, 
    #     0.12, 
    #     0.2, 
    #     0.15, 
    #     0.25, 
    #     0.1, 
    #     0.12, 
    #     0.14
    # ]
    # }


# 六足仿生——类昆虫式多足爬行机器人
    # json_param = {
    # "parameters": [
    #     0.12, 
    #     0.3, 
    #     0.45, 
    #     0.2, 
    #     0.3, 
    #     0.18, 
    #     0.28, 
    #     0.07, 
    #     0.09, 
    #     0.1
    # ]
    # }

# 设计理念：装甲碾压型——重载压步机器人
    # json_param = {
    # "parameters": [
    #     0.35, 
    #     0.1, 
    #     0.2, 
    #     0.25, 
    #     0.35, 
    #     0.3, 
    #     0.4, 
    #     0.12, 
    #     0.15, 
    #     0.18
    # ]
    # }

# # 极端适应——变形灵活型仿生机器人
    # json_param = {
    # "parameters": [
    #     0.15, 
    #     0.35, 
    #     0.5, 
    #     0.4, 
    #     0.6, 
    #     0.5, 
    #     0.7, 
    #     0.03, 
    #     0.05, 
    #     0.06
    # ]
    # }


# 浮游多足——反重力悬浮步行机器人
    # json_param = {
    # "parameters": [
    #     0.2, 
    #     0.4, 
    #     0.6, 
    #     0.5, 
    #     0.7, 
    #     0.6, 
    #     0.8, 
    #     0.02, 
    #     0.03, 
    #     0.04
    # ]
    # }

        # human expert: 
    # json_param = {
    # "parameters": [
    #     0.25, 
    #     0.2, 
    #     0.2, 
    #     0.2, 
    #     0.2, 
    #     0.4, 
    #     0.4, 
    #     0.08, 
    #     0.08, 
    #     0.08
    # ]
    # }   


    parameter = json_param["parameters"]
    volume = compute_ant_volume(parameter)
    print(f"Ant Volume = {volume:.6f}")
