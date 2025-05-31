import numpy as np
import os
import json
import math
import re
import random

# 计算体积
def calculate_capsule_volume(radius, height):
    """
    Calculate the volume of a capsule given the radius and height.
    
    Parameters:
    - radius (float): The radius of the capsule.
    - height (float): The height of the capsule.
    
    Returns:
    - volume (float): The volume of the capsule.
    """
    return math.pi * radius**2 * (height + (4 * radius / 3))
def compute_walker_volume(params):
    """
    Compute the total volume of the walker robot model based on the given params.
    
    Parameters:
    - params (list of floats): The input parameters that represent sizes and positions.
    
    Returns:
    - total_volume (float): The total volume consumed by the walker.
    """
    # Assign sizes (radii) for torso, thigh, leg, and foot
    sizes = [params[7], params[8], params[9]]  # These represent sizes {8}, {9}, {10}, etc.

    # Updated heights based on the positional differences (fromto)
    heights_updated = [
        abs(params[1] - params[2]),  # Thigh: |2 - 3|
        abs(params[2] - params[3]),  # Leg: |3 - 4|
        abs(params[4] - params[5])   # Foot: |5 - 6|
    ]

    # Calculate volumes with the updated heights
    volumes_updated = []
    torso_volume = calculate_capsule_volume(params[6], abs(params[0] - params[1]))

    for size, height in zip(sizes, heights_updated):
        volume = calculate_capsule_volume(size, height)
        volumes_updated.append(volume)


    # Return the sum of all volumes
    other_volume = sum(volumes_updated)
    total_volume = torso_volume + 2*other_volume
    return total_volume
def compute_hopper_volume(params):
    """
    Compute the total volume of the hopper robot model based on the given params.
    
    Parameters:
    - params (list of floats): The input parameters that represent sizes and positions.
    
    Returns:
    - total_volume (float): The total volume consumed by the hopper.
    """
    # Assign sizes (radii) for torso, thigh, leg, and foot
    sizes = [params[6], params[7], params[8], params[9]]  # These represent sizes {8}, {9}, {10}, etc.

    # Updated heights based on the positional differences (fromto)
    heights_updated = [
        abs(params[0] - params[1]),  # Torso: |1 - 2|
        abs(params[1] - params[2]),  # Thigh: |2 - 3|
        abs(params[2] - params[3]),  # Leg: |3 - 4|
        abs(params[4] - params[5])   # Foot: |5 - 6|
    ]

    # Calculate volumes with the updated heights
    volumes_updated = []
    for size, height in zip(sizes, heights_updated):
        volume = calculate_capsule_volume(size, height)
        volumes_updated.append(volume)

    # Return the sum of all volumes
    total_volume = sum(volumes_updated)
    return total_volume
def compute_swimmer_volume(params):

    sizes = [params[3], params[4], params[5]]  

    # Updated heights based on the positional differences (fromto)
    heights_updated = [
        abs(params[0]),  # 
        abs(params[1]),  # 
        abs(params[2]),  # 
    ]

    # Calculate volumes with the updated heights
    volumes_updated = []
    for size, height in zip(sizes, heights_updated):
        volume = calculate_capsule_volume(size, height)
        volumes_updated.append(volume)

    # Return the sum of all volumes
    total_volume = sum(volumes_updated)
    return total_volume
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
    print(f"params: {params}")

    torso_r =  params[0]

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
def compute_cheetah_volume(params):

    # Extract radii for all body parts
    radii = params[16:24]  
    # Calculate heights (lengths) for all body parts
    heights = [
        abs(params[0] - params[1]),    # Torso length
        np.sqrt((params[2])**2 + (params[3])**2),  # Head length
        np.sqrt((params[4])**2 + (params[5])**2),    # Back thigh length
        np.sqrt((params[6])**2 + (params[7])**2),    # Back shin length
        np.sqrt((params[8])**2 + (params[9])**2),    # Back foot length
        np.sqrt((params[10])**2 + (params[11])**2),    # Front thigh length
        np.sqrt((params[12])**2 + (params[13])**2), # Front shin length
        np.sqrt((params[14])**2 + (params[15])**2),   # Front foot length
    ]
    # Calculate volume for each body part
    volumes = []
    for radius, height in zip(radii, heights):
        volume = calculate_capsule_volume(radius, height)
        volumes.append(volume)

    # Sum all volumes to get total volume
    total_volume = sum(volumes)
    return total_volume


def ant_design_powered(parameter):
    num_param = len(parameter)
    ant_xml = '''
<mujoco model="ant">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option integrator="RK4" timestep="0.01"/>
  <custom>
    <numeric data="0.0 0.0 0.8711102550927979 1.0 0.0 0.0 0.0 0.0 1.0 0.0 -1.0 0.0 -1.0 0.0 1.0" name="init_qpos"/>
  </custom>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom conaffinity="0" condim="3" density="5.0" friction="1 0.5 0.5" margin="0.01" rgba="0.8 0.6 0.4 1"/>
  </default>
  <asset>
    <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0.9 0.9 0.9" type="skybox" width="100"/>
    <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
    <texture builtin="checker" height="100" name="texplane" rgb1="1 1 1" rgb2="0.7 0.7 0.7" type="2d" width="100"/>
    <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
    <material name="geom" texture="texgeom" texuniform="true"/>
  </asset>
  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom conaffinity="1" condim="3" material="MatPlane" name="floor" pos="0 0 0" rgba="0.95 0.95 0.95 1" size="40 40 40" type="plane"/>
    <body name="torso" pos="0 0 {height}">
      <camera name="track" mode="trackcom" pos="0 -3 0.3" xyaxes="1 0 0 0 0 1"/>
      <geom name="torso_geom" pos="0 0 0" size="{0}" type="sphere" rgba="0.47 0.73 0.87 1.0"/>
      <joint armature="0" damping="0" limited="false" margin="0.01" name="root" pos="0 0 0" type="free"/>
      <body name="front_left_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 {1} {2} 0.0" name="aux_1_geom" size="{7}" type="capsule" rgba="0.47 0.73 0.87 1.0"/>
        <body name="aux_1" pos="{1} {2} 0">
          <joint axis="0 0 1" name="hip_1" pos="0.0 0.0 0.0" range="-40 40" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 {3} {4} 0.0" name="left_leg_geom" size="{8}" type="capsule" rgba="0.47 0.73 0.87 1.0" />
          <body pos="{3} {4} 0" >
            <joint axis="-1 1 0" name="ankle_1" pos="0.0 0.0 0.0" range="30 100" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 {5} {6} 0.0" name="left_ankle_geom" size="{9}" type="capsule" rgba="0.47 0.73 0.87 1.0" />
          </body>
        </body>
      </body>
      <body name="front_right_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 -{1} {2} 0.0" name="aux_2_geom" size="{7}" type="capsule" rgba="0.47 0.73 0.87 1.0" />
        <body name="aux_2" pos="-{1} {2} 0">
          <joint axis="0 0 1" name="hip_2" pos="0.0 0.0 0.0" range="-40 40" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 -{3} {4} 0.0" name="right_leg_geom" size="{8}" type="capsule" rgba="0.47 0.73 0.87 1.0" />
          <body pos="-{3} {4} 0" >
            <joint axis="1 1 0" name="ankle_2" pos="0.0 0.0 0.0" range="-100 -30" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 -{5} {6} 0.0" name="right_ankle_geom" size="{9}" type="capsule" rgba="0.47 0.73 0.87 1.0" />
          </body>
        </body>
      </body>
      <body name="left_back_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 -{1} -{2} 0.0" name="aux_3_geom" size="{7}" type="capsule" rgba="0.47 0.73 0.87 1.0" />
        <body name="aux_3" pos="-{1} -{2} 0">
          <joint axis="0 0 1" name="hip_3" pos="0.0 0.0 0.0" range="-40 40" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 -{3} -{4} 0.0" name="back_leg_geom" size="{8}" type="capsule" rgba="0.47 0.73 0.87 1.0" />
          <body pos="-{3} -{4} 0" >
            <joint axis="-1 1 0" name="ankle_3" pos="0.0 0.0 0.0" range="-100 -30" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 -{5} -{6} 0.0" name="third_ankle_geom" size="{9}" type="capsule" rgba="0.47 0.73 0.87 1.0" />
          </body>
        </body>
      </body>
      <body name="right_back_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 {1} -{2} 0.0" name="aux_4_geom" size="{7}" type="capsule" rgba="0.47 0.73 0.87 1.0" />
        <body name="aux_4" pos=" {1} -{2} 0">
          <joint axis="0 0 1" name="hip_4" pos="0.0 0.0 0.0" range="-40 40" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 {3} -{4} 0.0" name="rightback_leg_geom" size="{8}" type="capsule" rgba="0.47 0.73 0.87 1.0" />
          <body pos="{3} -{4} 0" >
            <joint axis="1 1 0" name="ankle_4" pos="0.0 0.0 0.0" range="30 100" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 {5} -{6} 0.0" name="fourth_ankle_geom" size="{9}" type="capsule" rgba="0.47 0.73 0.87 1.0" />
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_4" gear="300"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_4" gear="300"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_1" gear="300"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_1" gear="300"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_2" gear="300"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_2" gear="300"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_3" gear="300"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_3" gear="300"/>
  </actuator>
</mujoco>
'''

    for i in range(num_param):
      ant_xml = ant_xml.replace("{"+str(i)+"}", str(parameter[i])) 
    z_height = math.sqrt(parameter[5]**2 + parameter[6]**2) + 0.15

    # 使用正则表达式修改 torso 的 pos
    ant_xml = ant_xml.replace("{height}", str(z_height)) 

    return ant_xml

def ant_design_desert(parameter):
    
    num_param = len(parameter)
    ant_xml = '''

<mujoco model="ant">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option integrator="RK4" timestep="0.01"/>
  <visual>
  <global offwidth="3840" offheight="2160"/>
  <map shadowscale=".05" />
  <quality shadowsize ="10000" />
  </visual>
  <custom>
    <numeric data="0.0 0.0 0.8711102550927979 1.0 0.0 0.0 0.0 0.0 1.0 0.0 -1.0 0.0 -1.0 0.0 1.0" name="init_qpos"/>
  </custom>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom conaffinity="0" condim="3" density="5.0" friction="1 0.5 0.5" margin="0.01" rgba="0.8 0.6 0.4 1"/>
  </default>
  <asset>
    <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0.9 0.9 0.9" type="skybox" width="100"/>
    <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
    <texture name="tex_desert" builtin="flat" type="2d" width="128" height="128" rgb1="0.95 0.85 0.65" rgb2="0.9 0.8 0.6" mark="none" />
    <material name="mat_desert" texture="tex_desert" rgba="1 1 1 1" specular="0.01" shininess="0.1" reflectance="0.2" />
    <hfield name="floor_desert" file="desert.png" size="32 32 1.5 0.1"/>
  </asset>

  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1" directional="true" exponent="10" pos="0 0 5" specular=".1 .1 .1"  castshadow="false" />
    <light cutoff="200" diffuse=".3 .3 .3" dir="-1 -1 0" directional="true" exponent=".1" pos="3 3 0" specular=".1 .1 .1"  castshadow="true" />
    <geom name='floor1' material="mat_desert" pos='17 0 -1.5' type='hfield' conaffinity='1' rgba="1 1 1 1" condim='3' hfield="floor_desert" friction="0.4 0.005 0.0001" />
    <geom name='floor2' material="mat_desert" pos='81 0 -1.5' type='hfield' conaffinity='1' rgba="1 1 1 1" condim='3' hfield="floor_desert" friction="0.4 0.005 0.0001" />

    <body name="torso" pos="0 0 {height}">
      <camera name="track" mode="trackcom" pos="0 -3 0.3" xyaxes="1 0 0 0 0 1"/>
      <geom name="torso_geom" pos="0 0 0" size="{0}" type="sphere"/>
      <joint armature="0" damping="0" limited="false" margin="0.01" name="root" pos="0 0 0" type="free"/>
      <body name="front_left_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 {1} {2} 0.0" name="aux_1_geom" size="{7}" type="capsule"/>
        <body name="aux_1" pos="{1} {2} 0">
          <joint axis="0 0 1" name="hip_1" pos="0.0 0.0 0.0" range="-40 40" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 {3} {4} 0.0" name="left_leg_geom" size="{8}" type="capsule" />
          <body pos="{3} {4} 0" >
            <joint axis="-1 1 0" name="ankle_1" pos="0.0 0.0 0.0" range="30 100" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 {5} {6} 0.0" name="left_ankle_geom" size="{9}" type="capsule"/>
          </body>
        </body>
      </body>
      <body name="front_right_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 -{1} {2} 0.0" name="aux_2_geom" size="{7}" type="capsule"/>
        <body name="aux_2" pos="-{1} {2} 0">
          <joint axis="0 0 1" name="hip_2" pos="0.0 0.0 0.0" range="-40 40" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 -{3} {4} 0.0" name="right_leg_geom" size="{8}" type="capsule"/>
          <body pos="-{3} {4} 0" >
            <joint axis="1 1 0" name="ankle_2" pos="0.0 0.0 0.0" range="-100 -30" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 -{5} {6} 0.0" name="right_ankle_geom" size="{9}" type="capsule"/>
          </body>
        </body>
      </body>
      <body name="left_back_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 -{1} -{2} 0.0" name="aux_3_geom" size="{7}" type="capsule"/>
        <body name="aux_3" pos="-{1} -{2} 0">
          <joint axis="0 0 1" name="hip_3" pos="0.0 0.0 0.0" range="-40 40" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 -{3} -{4} 0.0" name="back_leg_geom" size="{8}" type="capsule"/>
          <body pos="-{3} -{4} 0" >
            <joint axis="-1 1 0" name="ankle_3" pos="0.0 0.0 0.0" range="-100 -30" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 -{5} -{6} 0.0" name="third_ankle_geom" size="{9}" type="capsule"/>
          </body>
        </body>
      </body>
      <body name="right_back_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 {1} -{2} 0.0" name="aux_4_geom" size="{7}" type="capsule"/>
        <body name="aux_4" pos=" {1} -{2} 0">
          <joint axis="0 0 1" name="hip_4" pos="0.0 0.0 0.0" range="-40 40" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 {3} -{4} 0.0" name="rightback_leg_geom" size="{8}" type="capsule"/>
          <body pos="{3} -{4} 0" >
            <joint axis="1 1 0" name="ankle_4" pos="0.0 0.0 0.0" range="30 100" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 {5} -{6} 0.0" name="fourth_ankle_geom" size="{9}" type="capsule"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_4" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_4" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_1" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_1" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_2" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_2" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_3" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_3" gear="150"/>
  </actuator>
</mujoco>
'''

    for i in range(num_param):
      ant_xml = ant_xml.replace("{"+str(i)+"}", str(parameter[i])) 
    z_height = math.sqrt(parameter[5]**2 + parameter[6]**2) + 0.15

    # 使用正则表达式修改 torso 的 pos
    ant_xml = ant_xml.replace("{height}", str(z_height)) 

    return ant_xml

# jump ant 和 ant用的同一个
def ant_design(parameter):
    num_param = len(parameter)
    ant_xml = '''
<mujoco model="ant">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option integrator="RK4" timestep="0.01"/>
  <custom>
    <numeric data="0.0 0.0 0.8711102550927979 1.0 0.0 0.0 0.0 0.0 1.0 0.0 -1.0 0.0 -1.0 0.0 1.0" name="init_qpos"/>
  </custom>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom conaffinity="0" condim="3" density="5.0" friction="1 0.5 0.5" margin="0.01" rgba="0.8 0.6 0.4 1"/>
  </default>
  <asset>
    <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0.9 0.9 0.9" type="skybox" width="100"/>
    <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
    <texture builtin="checker" height="100" name="texplane" rgb1="1 1 1" rgb2="0.7 0.7 0.7" type="2d" width="100"/>
    <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
    <material name="geom" texture="texgeom" texuniform="true"/>
  </asset>
  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom conaffinity="1" condim="3" material="MatPlane" name="floor" pos="0 0 0" rgba="0.95 0.95 0.95 1" size="40 40 40" type="plane"/>
    <body name="torso" pos="0 0 {height}">
      <camera name="track" mode="trackcom" pos="0 -3 0.3" xyaxes="1 0 0 0 0 1"/>
      <geom name="torso_geom" pos="0 0 0" size="{0}" type="sphere" rgba="0.47 0.73 0.87 1.0"/>
      <joint armature="0" damping="0" limited="false" margin="0.01" name="root" pos="0 0 0" type="free"/>
      <body name="front_left_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 {1} {2} 0.0" name="aux_1_geom" size="{7}" type="capsule" rgba="0.47 0.73 0.87 1.0"/>
        <body name="aux_1" pos="{1} {2} 0">
          <joint axis="0 0 1" name="hip_1" pos="0.0 0.0 0.0" range="-40 40" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 {3} {4} 0.0" name="left_leg_geom" size="{8}" type="capsule" rgba="0.47 0.73 0.87 1.0" />
          <body pos="{3} {4} 0" >
            <joint axis="-1 1 0" name="ankle_1" pos="0.0 0.0 0.0" range="30 100" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 {5} {6} 0.0" name="left_ankle_geom" size="{9}" type="capsule" rgba="0.47 0.73 0.87 1.0" />
          </body>
        </body>
      </body>
      <body name="front_right_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 -{1} {2} 0.0" name="aux_2_geom" size="{7}" type="capsule" rgba="0.47 0.73 0.87 1.0" />
        <body name="aux_2" pos="-{1} {2} 0">
          <joint axis="0 0 1" name="hip_2" pos="0.0 0.0 0.0" range="-40 40" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 -{3} {4} 0.0" name="right_leg_geom" size="{8}" type="capsule" rgba="0.47 0.73 0.87 1.0" />
          <body pos="-{3} {4} 0" >
            <joint axis="1 1 0" name="ankle_2" pos="0.0 0.0 0.0" range="-100 -30" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 -{5} {6} 0.0" name="right_ankle_geom" size="{9}" type="capsule" rgba="0.47 0.73 0.87 1.0" />
          </body>
        </body>
      </body>
      <body name="left_back_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 -{1} -{2} 0.0" name="aux_3_geom" size="{7}" type="capsule" rgba="0.47 0.73 0.87 1.0" />
        <body name="aux_3" pos="-{1} -{2} 0">
          <joint axis="0 0 1" name="hip_3" pos="0.0 0.0 0.0" range="-40 40" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 -{3} -{4} 0.0" name="back_leg_geom" size="{8}" type="capsule" rgba="0.47 0.73 0.87 1.0" />
          <body pos="-{3} -{4} 0" >
            <joint axis="-1 1 0" name="ankle_3" pos="0.0 0.0 0.0" range="-100 -30" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 -{5} -{6} 0.0" name="third_ankle_geom" size="{9}" type="capsule" rgba="0.47 0.73 0.87 1.0" />
          </body>
        </body>
      </body>
      <body name="right_back_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 {1} -{2} 0.0" name="aux_4_geom" size="{7}" type="capsule" rgba="0.47 0.73 0.87 1.0" />
        <body name="aux_4" pos=" {1} -{2} 0">
          <joint axis="0 0 1" name="hip_4" pos="0.0 0.0 0.0" range="-40 40" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 {3} -{4} 0.0" name="rightback_leg_geom" size="{8}" type="capsule" rgba="0.47 0.73 0.87 1.0" />
          <body pos="{3} -{4} 0" >
            <joint axis="1 1 0" name="ankle_4" pos="0.0 0.0 0.0" range="30 100" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 {5} -{6} 0.0" name="fourth_ankle_geom" size="{9}" type="capsule" rgba="0.47 0.73 0.87 1.0" />
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_4" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_4" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_1" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_1" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_2" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_2" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_3" gear="150"/>
    <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_3" gear="150"/>
  </actuator>
</mujoco>
'''

    for i in range(num_param):
      ant_xml = ant_xml.replace("{"+str(i)+"}", str(parameter[i])) 
    z_height = math.sqrt(parameter[5]**2 + parameter[6]**2) + 0.15

    # 使用正则表达式修改 torso 的 pos
    ant_xml = ant_xml.replace("{height}", str(z_height)) 

    return ant_xml




def hopper_design(parameter):
  # parameter = [1.45, 1.06, 0.6, 0.1, -0.13, 0.26, 0.05, 0.05, 0.04, 0.06]
  # 标准参数
  # filename = "GPTHopper.xml" 
  num_param = len(parameter)

  param_xml = '''
  <mujoco model="hopper">
    <compiler angle="degree" coordinate="global" inertiafromgeom="true"/>
    <default>
      <joint armature="1" damping="1" limited="true"/>
      <geom conaffinity="1" condim="1" contype="1" margin="0.001" material="geom" rgba="0.8 0.6 .4 1" solimp=".8 .8 .01" solref=".02 1"/>
      <motor ctrllimited="true" ctrlrange="-.4 .4"/>
    </default>
    <asset>
      <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0.9 0.9 0.9" type="skybox" width="100"/>
      <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
      <texture builtin="checker" height="100" name="texplane" rgb1="1 1 1" rgb2="0.7 0.7 0.7" type="2d" width="100"/>
      <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
      <material name="geom" texture="texgeom" texuniform="true"/>
    </asset>
    <option integrator="RK4" timestep="0.002"/>

    <worldbody>
      <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
      <geom conaffinity="1" condim="3" material="MatPlane" name="floor" pos="0 0 0" rgba="0.95 0.95 0.95 1" size="20 20 .125" type="plane"/>
      <body name="torso">
        <camera name="track" mode="trackcom" pos="0 -3 -0.25" xyaxes="1 0 0 0 0 1"/>
        <joint armature="0" axis="1 0 0" damping="0" limited="false" name="ignore1" pos="0 0 0" stiffness="0" type="slide"/>
        <joint armature="0" axis="0 0 1" damping="0" limited="false" name="ignore2" pos="0 0 0" ref="1.25" stiffness="0" type="slide"/>
        <joint armature="0" axis="0 1 0" damping="0" limited="false" name="ignore3" pos="0 0 0" stiffness="0" type="hinge"/>
        <geom fromto="0 0 {1} 0 0 {2}" name="torso_geom" size="{8}" type="capsule" friction="0.9"/>
        <body name="thigh">
          <joint axis="0 -1 0" name="thigh_joint" pos="0 0 {2}" range="-150 0" type="hinge"/>
          <geom fromto="0 0 {2} 0 0 {3}" name="thigh_geom" size="{8}" type="capsule" friction="0.9" rgba="0.8 0.6 0.4 1" />
          <body name="leg">
            <joint axis="0 -1 0" name="leg_joint" pos="0 0 {3}" range="-150 0" type="hinge"/>
            <geom fromto="0 0 {3} 0 0 {4}" name="leg_geom" size="{9}" type="capsule" friction="0.9" rgba="0.8 0.6 0.4 1" />
            <body name="foot">
              <joint axis="0 -1 0" name="foot_joint" pos="0 0 {4}" range="-45 45" type="hinge"/>
              <geom fromto="{5} 0 {4} {6} 0 {4}" name="foot_geom" size="{10}" type="capsule" friction="2.0" rgba="0.8 0.6 0.4 1" />
            </body>
          </body>
        </body>
      </body>
    </worldbody>
    <actuator>
      <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="thigh_joint"/>
      <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="leg_joint"/>
      <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="200.0" joint="foot_joint"/>
    </actuator>
      <asset>
        <texture type="skybox" builtin="gradient" rgb1=".4 .5 .6" rgb2="0 0 0"
            width="100" height="100"/>
        <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
        <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
        <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
        <material name="geom" texture="texgeom" texuniform="true"/>
      </asset>
  </mujoco>
  '''
  for i in range(num_param):
    param_xml = param_xml.replace("{"+str(i+1)+"}", str(parameter[i]))

  return param_xml

def walker_design(parameter):
  # parameter = [1.45, 1.06, 0.6, 0.1, -0.13, 0.26, 0.05, 0.05, 0.04, 0.06]
  # 标准参数
  num_param = len(parameter)

  param_xml = '''
  <mujoco model="walker2d">
    <compiler angle="degree" coordinate="global" inertiafromgeom="true"/>
    <default>
      <joint armature="0.01" damping=".1" limited="true"/>
      <geom conaffinity="0" condim="3" contype="1" density="1000" friction=".7 .1 .1" rgba="0.8 0.6 .4 1"/>
    </default>
    <option integrator="RK4" timestep="0.002"/>

    <worldbody>
      <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
      <geom conaffinity="1" condim="3" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="20 20 .125" type="plane" material="MatPlane"/>

      <body name="torso">
        <camera name="track" mode="trackcom" pos="0 -3 -0.25" xyaxes="1 0 0 0 0 1"/>
        <joint armature="0" axis="1 0 0" damping="0" limited="false" name="ignore1" pos="0 0 0" stiffness="0" type="slide"/>
        <joint armature="0" axis="0 0 1" damping="0" limited="false" name="ignore2" pos="0 0 0" ref="1.25" stiffness="0" type="slide"/>
        <joint armature="0" axis="0 1 0" damping="0" limited="false" name="ignore3" pos="0 0 0" stiffness="0" type="hinge"/>
        <geom fromto="0 0 {1} 0 0 {2}" name="torso_geom" size="{7}" type="capsule" friction="0.9"/>
        <body name="thigh">
          <joint axis="0 -1 0" name="thigh_joint" pos="0 0 {2}" range="-150 0" type="hinge"/>
          <geom fromto="0 0 {2} 0 0 {3}" name="thigh_geom" size="{8}" type="capsule" friction="0.9"/>
          <body name="leg">
            <joint axis="0 -1 0" name="leg_joint" pos="0 0 {3}" range="-150 0" type="hinge"/>
            <geom fromto="0 0 {3} 0 0 {4}" name="leg_geom" size="{9}" type="capsule" friction="0.9"/>
            <body name="foot">
              <joint axis="0 -1 0" name="foot_joint" pos="0 0 {4}" range="-45 45" type="hinge"/>
              <geom fromto="{5} 0 {4} {6} 0 {4}" name="foot_geom" size="{10}" type="capsule" friction="1.9"/>
            </body>
          </body>
        </body>
        <body name="thigh_left">
          <joint axis="0 -1 0" name="thigh_left_joint" pos="0 0 {2}" range="-150 0" type="hinge"/>
          <geom fromto="0 0 {2} 0 0 {3}" name="thigh_left_geom" size="{8}" type="capsule" friction="0.9"/>
          <body name="leg_left">
            <joint axis="0 -1 0" name="leg_left_joint" pos="0 0 {3}" range="-150 0" type="hinge"/>
            <geom fromto="0 0 {3} 0 0 {4}" name="leg_left_geom" size="{9}" type="capsule" friction="0.9"/>
            <body name="foot_left">
              <joint axis="0 -1 0" name="foot_left_joint" pos="0 0 {4}" range="-45 45" type="hinge"/>
              <geom fromto="{5} 0 {4} {6} 0 {4}" name="foot_left_geom" size="{10}" type="capsule" friction="1.9"/>
            </body>
          </body>
        </body>
      </body>
    </worldbody>

    <actuator>
      <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="100" joint="thigh_joint"/>
      <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="100" joint="leg_joint"/>
      <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="100" joint="foot_joint"/>
      <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="100" joint="thigh_left_joint"/>
      <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="100" joint="leg_left_joint"/>
      <motor ctrllimited="true" ctrlrange="-1.0 1.0" gear="100" joint="foot_left_joint"/>
    </actuator>
      <asset>
        <texture type="skybox" builtin="gradient" rgb1=".4 .5 .6" rgb2="0 0 0"
            width="100" height="100"/>
        <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
        <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
        <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
        <material name="geom" texture="texgeom" texuniform="true"/>
      </asset>
  </mujoco>
  '''


  for i in range(num_param):
    param_xml = param_xml.replace("{"+str(i+1)+"}", str(parameter[i]))

  return param_xml
  
def cheetah_design2(parameter):
    
    print("executing half cheetah design")
    # Extract the torso endpoints from the parameter list
    torso_back = parameter[0]    # Negative value, e.g. -0.5
    torso_front = parameter[1]   # Positive value, e.g. 0.5

    # Calculate head position based on torso length
    # This places the head a bit beyond the front of the torso
    head_x = torso_front + 0.1   # 0.1 is the offset beyond the front of the torso
    head_z = 0.1                 # Keep the height constant

    # Update the parameters for the head position
    parameter[2] = head_x
    parameter[3] = head_z

    # The rest remains the same
    # parameter = [torso_back, torso_front, head_x, head_z, 0.1, -0.13, 0.16, -0.25, -0.14, -0.07, -0.28, -0.14, 0.03, -0.097,
    # -0.07, -0.12, -0.14, -0.24, 0.065, -0.09, 0.13, -0.18, 0.045, -0.07,
    # 0.046, 0.046, 0.15, 0.046, 0.145, 0.046, 0.15, 0.046, 0.094, 0.046, 0.133, 0.046, 0.106, 0.046, 0.07]


    # # 标准参数
    # filename = "GPTHalfCheetah.xml" 
    num_param = len(parameter)


    param_xml = '''
  <mujoco model="cheetah">
    <compiler angle="radian" coordinate="local" inertiafromgeom="true" settotalmass="14"/>
    <default>
      <joint armature=".1" damping=".01" limited="true" solimplimit="0 .8 .03" solreflimit=".02 1" stiffness="8"/>
      <geom conaffinity="0" condim="3" contype="1" friction=".4 .1 .1" rgba="0.8 0.6 .4 1" solimp="0.0 0.8 0.01" solref="0.02 1"/>
      <motor ctrllimited="true" ctrlrange="-1 1"/>
    </default>
    <size nstack="300000" nuser_geom="1"/>
    <option gravity="0 0 -9.81" timestep="0.01"/>
    <asset>
      <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>
      <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
      <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
      <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
      <material name="geom" texture="texgeom" texuniform="true"/>
    </asset>
    <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom conaffinity="1" condim="3" material="MatPlane" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="40 40 40" type="plane"/>
    <body name="torso" pos="0 0 .7">
      <joint armature="0" axis="1 0 0" damping="0" limited="false" name="ignorex" pos="0 0 0" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 0 1" damping="0" limited="false" name="ignorez" pos="0 0 0" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 1 0" damping="0" limited="false" name="ignorey" pos="0 0 0" stiffness="0" type="hinge"/>

      <geom fromto="{0} 0 0 {1} 0 0" name="torso" size="{24}" type="capsule"/>
      <geom axisangle="0 1 0 .87" name="head" pos="{2} 0 {3}" size="{25} {26}" type="capsule"/>

      <body name="bthigh" pos="{0} 0 0">
        <joint axis="0 1 0" damping="6" name="bthigh" pos="0 0 0" range="-.52 1.05" stiffness="240" type="hinge"/>
        <geom axisangle="0 1 0 -3.8" name="bthigh" pos="{4} 0 {5}" size="{27} {28}" type="capsule"/>

        <body name="bshin" pos="{6} 0 {7}">
          <joint axis="0 1 0" damping="4.5" name="bshin" pos="0 0 0" range="-.785 .785" stiffness="180" type="hinge"/>
          <geom axisangle="0 1 0 -2.03" name="bshin" pos="{8} 0 {9}" rgba="0.9 0.6 0.6 1" size="{29} {30}" type="capsule"/>
          <body name="bfoot" pos="{10} 0 {11}">
            <joint axis="0 1 0" damping="3" name="bfoot" pos="0 0 0" range="-.4 .785" stiffness="120" type="hinge"/>
            <geom axisangle="0 1 0 -.27" name="bfoot" pos="{12} 0 {13}" rgba="0.9 0.6 0.6 1" size="{31} {32}" type="capsule"/>

          </body>
        </body>
      </body>

      <body name="fthigh" pos="{1} 0 0">
        <joint axis="0 1 0" damping="4.5" name="fthigh" pos="0 0 0" range="-1.5 0.8" stiffness="180" type="hinge"/>
        <geom axisangle="0 1 0 .52" name="fthigh" pos="{14} 0 {15}" size="{33} {34}" type="capsule"/>
        <body name="fshin" pos="{16} 0 {17}">
          <joint axis="0 1 0" damping="3" name="fshin" pos="0 0 0" range="-1.2 1.1" stiffness="120" type="hinge"/>
          <geom axisangle="0 1 0 -.6" name="fshin" pos="{18} 0 {19}" rgba="0.9 0.6 0.6 1" size="{35} {36}" type="capsule"/>
          <body name="ffoot" pos="{20} 0 {21}">
            <joint axis="0 1 0" damping="1.5" name="ffoot" pos="0 0 0" range="-3.1 -0.3" stiffness="60" type="hinge"/>
            <geom axisangle="0 1 0 -.6" name="ffoot" pos="{22} 0 {23}" rgba="0.9 0.6 0.6 1" size="{37} {38}" type="capsule"/>

          </body>
        </body>
      </body>
    </body>

    </worldbody>
    <actuator>
    <motor gear="120" joint="bthigh" name="bthigh"/>
    <motor gear="90" joint="bshin" name="bshin"/>
    <motor gear="60" joint="bfoot" name="bfoot"/>
    <motor gear="120" joint="fthigh" name="fthigh"/>
    <motor gear="60" joint="fshin" name="fshin"/>
    <motor gear="30" joint="ffoot" name="ffoot"/>
    </actuator>
    </mujoco>
    '''
    
    print("Replacing half cheetah parameters")
    
    for i in range(num_param):
        param_xml = param_xml.replace("{"+str(i)+"}", str(parameter[i]))

    print("halfcheetahdesign:", param_xml)
  
    fp = open("cheetah.xml", "w")
    fp.write(param_xml)
    fp.close()


    return param_xml
  



def swimmer_design(parameter):
  # parameter = [1, 2.5, 2, 0.2, 0.3, 0.4]

  # 标准参数
  filename = "GPTSwimmer.xml" 
  num_param = len(parameter)

  param_xml = '''
<mujoco model="swimmer">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option density="4000" integrator="RK4" timestep="0.01" viscosity="0.1"/>
  <default>
    <geom conaffinity="0" condim="1" contype="0" material="geom" rgba="0.8 0.6 .4 1"/>
    <joint armature='0.1'  />
  </default>
  <asset>
    <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>
    <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
    <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
    <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="30 30" texture="texplane"/>
    <material name="geom" texture="texgeom" texuniform="true"/>
  </asset>
  
  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom condim="3" material="MatPlane" name="floor" pos="0 0 -0.1" rgba="0.8 0.9 0.8 1" size="40 40 0.1" type="plane"/>
    <!--  ================= SWIMMER ================= /-->
    <body name="torso" pos="0 0 0">
      <camera name="track" mode="trackcom" pos="0 -3 3" xyaxes="1 0 0 0 1 1"/>
      <geom density="1000" fromto="{1} 0 0 0 0 0" size="{4}" type="capsule"/>
      <joint axis="1 0 0" name="slider1" pos="0 0 0" type="slide"/>
      <joint axis="0 1 0" name="slider2" pos="0 0 0" type="slide"/>
      <joint axis="0 0 1" name="free_body_rot" pos="0 0 0" type="hinge"/>
      <body name="mid" pos="0 0 0">
        <geom density="1000" fromto="0 0 0 -{2} 0 0" size="{5}" type="capsule"/>
        <joint axis="0 0 1" limited="true" name="motor1_rot" pos="0 0 0" range="-100 100" type="hinge"/>
        <body name="back" pos="-{2} 0 0">
          <geom density="1000" fromto="0 0 0 -{3} 0 0" size="{6}" type="capsule"/>
          <joint axis="0 0 1" limited="true" name="motor2_rot" pos="0 0 0" range="-100 100" type="hinge"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor ctrllimited="true" ctrlrange="-1 1" gear="200.0" joint="motor1_rot"/>
    <motor ctrllimited="true" ctrlrange="-1 1" gear="200.0" joint="motor2_rot"/>
  </actuator>
</mujoco>

  '''
  for i in range(num_param):
    param_xml = param_xml.replace("{"+str(i+1)+"}", str(parameter[i]))

  # fp = open(os.path.join(os.path.dirname(__file__), filename), "w")
  # fp.write(param_xml)
  # fp.close()
  return param_xml
  

def cheetah_design(parameter):
    
    # print("executing half cheetah design")
    # # Extract the torso endpoints from the parameter list
    # torso_back = parameter[0]    # Negative value, e.g. -0.5
    # torso_front = parameter[1]   # Positive value, e.g. 0.5

    # # Calculate head position based on torso length
    # # This places the head a bit beyond the front of the torso
    # head_x = torso_front + 0.1   # 0.1 is the offset beyond the front of the torso
    # head_z = 0.1                 # Keep the height constant

    # # Update the parameters for the head position
    # parameter[2] = head_x
    # parameter[3] = head_z


    # # # 标准参数
    # # filename = "GPTHalfCheetah.xml" 

    num_param = len(parameter)

    param_xml = '''
  <mujoco model="cheetah">
    <compiler angle="radian" coordinate="local" inertiafromgeom="true" settotalmass="14"/>
    <default>
      <joint armature=".1" damping=".01" limited="true" solimplimit="0 .8 .03" solreflimit=".02 1" stiffness="8"/>
      <geom conaffinity="0" condim="3" contype="1" friction=".4 .1 .1" rgba="0.8 0.6 .4 1" solimp="0.0 0.8 0.01" solref="0.02 1"/>
      <motor ctrllimited="true" ctrlrange="-1 1"/>
    </default>
    <size nstack="300000" nuser_geom="1"/>
    <option gravity="0 0 -9.81" timestep="0.01"/>
    <asset>
      <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>
      <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
      <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
      <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
      <material name="geom" texture="texgeom" texuniform="true"/>
    </asset>
    <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom conaffinity="1" condim="3" material="MatPlane" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="40 40 40" type="plane"/>
    <body name="torso" pos="0 0 {height}">
      <joint armature="0" axis="1 0 0" damping="0" limited="false" name="ignorex" pos="0 0 0" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 0 1" damping="0" limited="false" name="ignorez" pos="0 0 0" stiffness="0" type="slide"/>
      <joint armature="0" axis="0 1 0" damping="0" limited="false" name="ignorey" pos="0 0 0" stiffness="0" type="hinge"/>
      <geom fromto="{param1} 0 0 {param2} 0 0" name="torso" size="{param17}" type="capsule"/>
      <geom fromto="{param2} 0 0 {param3} 0 {param4}" name="head"  size="{param18}" type="capsule"/>

      <body name="bthigh" pos="{param1} 0 0">
        <joint axis="0 1 0" damping="6" name="bthigh" pos="0 0 0" range="-.52 1.05" stiffness="240" type="hinge"/>
        <geom fromto = "0 0 0 {param5} 0 {param6}" name="bthigh" size="{param19}" type="capsule"/>
        <body name="bshin" pos="{param5} 0 {param6}">
          <joint axis="0 1 0" damping="4.5" name="bshin" pos="0 0 0" range="-.785 .785" stiffness="180" type="hinge"/>
          <geom fromto = "0 0 0 {param7} 0 {param8}" name="bshin" rgba="0.9 0.6 0.6 1" size="{param20}" type="capsule"/>
          <body name="bfoot" pos="{param7} 0 {param8}">
            <joint axis="0 1 0" damping="3" name="bfoot" pos="0 0 0" range="-.4 .785" stiffness="120" type="hinge"/>
            <geom fromto = "0 0 0 {param9} 0 {param10}" name="bfoot" rgba="0.9 0.6 0.6 1" size="{param21}" type="capsule"/>
          </body>
        </body>
      </body>

      <body name="fthigh" pos="{param2} 0 0">
        <joint axis="0 1 0" damping="4.5" name="fthigh" pos="0 0 0" range="-1.5 0.8" stiffness="180" type="hinge"/>
        <geom fromto = "0 0 0 {param11} 0 {param12}" name="fthigh" size="{param22}" type="capsule"/>
        <body name="fshin" pos = "{param11} 0 {param12}">
          <joint axis="0 1 0" damping="3" name="fshin" pos="0 0 0" range="-1.2 1.1" stiffness="120" type="hinge"/>
          <geom fromto = "0 0 0 {param13} 0 {param14}" rgba="0.9 0.6 0.6 1" size="{param23}" type="capsule"/>
          <body name="ffoot" pos="{param13} 0 {param14}">
            <joint axis="0 1 0" damping="1.5" name="ffoot" pos="0 0 0" range="-3.1 -0.3" stiffness="60" type="hinge"/>
            <geom fromto = "0 0 0 {param15} 0 {param16}" name="ffoot" rgba="0.9 0.6 0.6 1" size="{param24}" type="capsule"/>
          </body>
        </body>
      </body>
    </body>

    </worldbody>
    <actuator>
    <motor gear="120" joint="bthigh" name="bthigh"/>
    <motor gear="90" joint="bshin" name="bshin"/>
    <motor gear="60" joint="bfoot" name="bfoot"/>
    <motor gear="120" joint="fthigh" name="fthigh"/>
    <motor gear="60" joint="fshin" name="fshin"/>
    <motor gear="30" joint="ffoot" name="ffoot"/>
    </actuator>
    </mujoco>
    '''
    # print(parameter[0])

    
    for i in range(num_param):
        param_xml = param_xml.replace("{param"+str(i+1)+"}", str(parameter[i]))
    z1 = np.abs(parameter[5] + parameter[7] + parameter[9])
    z2 = np.abs(parameter[11] + parameter[13] + parameter[15])
    # print(z1)
    # print(z2)
    z_height = max(z1, z2)
    param_xml = param_xml.replace("{height}", str(z_height))

    # fp = open("cheetah.xml", "w")
    # fp.write(param_xml)
    # fp.close()


    return param_xml



