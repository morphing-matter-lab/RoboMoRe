import numpy as np
import os
import json
import math
import re
def ant_design(parameter):
    
    num_param = len(parameter)
    ant_xml = '''
<mujoco model="ant">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option integrator="RK4" timestep="0.01"/>
  <custom>
    <numeric data="0.0 0.0 {height} 1.0 0.0 0.0 0.0 0.0 1.0 0.0 -1.0 0.0 -1.0 0.0 1.0" name="init_qpos"/>
  </custom>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom conaffinity="0" condim="3" density="5.0" friction="1 0.5 0.5" margin="0.01" rgba="0.8 0.6 0.4 1"/>
  </default>
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
    <option integrator="RK4" timestep="0.002"/>

    <worldbody>
      <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
      <geom conaffinity="1" condim="3" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="20 20 .125" type="plane" material="MatPlane"/>
      <body name="torso">
        <camera name="track" mode="trackcom" pos="0 -3 -0.25" xyaxes="1 0 0 0 0 1"/>
        <joint armature="0" axis="1 0 0" damping="0" limited="false" name="ignore1" pos="0 0 0" stiffness="0" type="slide"/>
        <joint armature="0" axis="0 0 1" damping="0" limited="false" name="ignore2" pos="0 0 0" ref="1.25" stiffness="0" type="slide"/>
        <joint armature="0" axis="0 1 0" damping="0" limited="false" name="ignore3" pos="0 0 0" stiffness="0" type="hinge"/>
        <geom fromto="0 0 {1} 0 0 {2}" name="torso_geom" size="{8}" type="capsule" friction="0.9"/>
        <body name="thigh">
          <joint axis="0 -1 0" name="thigh_joint" pos="0 0 {2}" range="-150 0" type="hinge"/>
          <geom fromto="0 0 {2} 0 0 {3}" name="thigh_geom" size="{8}" type="capsule" friction="0.9"/>
          <body name="leg">
            <joint axis="0 -1 0" name="leg_joint" pos="0 0 {3}" range="-150 0" type="hinge"/>
            <geom fromto="0 0 {3} 0 0 {4}" name="leg_geom" size="{9}" type="capsule" friction="0.9"/>
            <body name="foot">
              <joint axis="0 -1 0" name="foot_joint" pos="0 0 {4}" range="-45 45" type="hinge"/>
              <geom fromto="{5} 0 {4} {6} 0 {4}" name="foot_geom" size="{10}" type="capsule" friction="2.0"/>
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
  
def half_cheetah_design(parameter):
  parameter = [-0.5, 0.5, 0.6, 0.1, 0.1, -0.13, 0.16, -0.25, -0.14, -0.07, -0.28, -0.14, 0.03, -0.097,
  -0.07, -0.12, -0.14, -0.24, 0.065, -0.09, 0.13, -0.18, 0.045, -0.07,
  0.046, 0.046, 0.15, 0.046, 0.145, 0.046, 0.15, 0.046, 0.094, 0.046, 0.133, 0.046, 0.106, 0.046, 0.07]

  # 标准参数
  filename = "GPTHalf_cheetah.xml" 
  num_param = len(parameter)

  param_xml = '''
<mujoco model="cheetah">
  <compiler angle="radian" coordinate="local" inertiafromgeom="true" settotalmass="14"/>
  <default>
    <joint armature=".1" damping=".01" limited="true" solimplimit="0 .8 .03" solreflimit=".02 1" stiffness="8"/>
    <geom conaffinity="0" condim="3" contype="1" friction="0.8 .1 .1" rgba="0.8 0.6 .4 1" solimp="0.0 0.8 0.01" solref="0.02 1"/>
    <motor ctrllimited="true" ctrlrange="-1 1"/>
  </default>
  <size nstack="300000" nuser_geom="1"/>
  <option gravity="0 0 -9.81" timestep="0.01"/>

  <worldbody>
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
            <inertial mass="10"/>
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
            <inertial mass="10"/>
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
  for i in range(num_param):
    param_xml = param_xml.replace("{"+str(i+1)+"}", str(parameter[i]))

  fp = open(os.path.join(os.path.dirname(__file__), "assets", filename), "w")
  fp.write(param_xml)
  fp.close()
  
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
  
# swimmer_design(parameter=None)

def humanoid_design(parameter):
  parameter = [1, 1, 1, 0.02, 0.02, 0.02]

  # 标准参数
  filename = "GPTHumanoid.xml" 
  num_param = len(parameter)

  param_xml = '''
<mujoco model="humanoid">
    <compiler angle="degree" inertiafromgeom="true"/>
    <default>
        <joint armature="1" damping="1" limited="true"/>
        <geom conaffinity="1" condim="1" contype="1" margin="0.001" material="geom" rgba="0.8 0.6 .4 1"/>
        <motor ctrllimited="true" ctrlrange="-.4 .4"/>
    </default>
    <option integrator="RK4" iterations="50" solver="PGS" timestep="0.003">
        <!-- <flags solverstat="enable" energy="enable"/>-->
    </option>
    <size nkey="5" nuser_geom="1"/>
    <visual>
        <map fogend="5" fogstart="3"/>
    </visual>
    <asset>
        <texture builtin="gradient" height="100" rgb1=".4 .5 .6" rgb2="0 0 0" type="skybox" width="100"/>
        <!-- <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>-->
        <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
        <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
        <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
        <material name="geom" texture="texgeom" texuniform="true"/>
    </asset>
    <worldbody>
        <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
        <geom condim="3" friction="1 .1 .1" material="MatPlane" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="20 20 0.125" type="plane"/>
        <!-- <geom condim="3" material="MatPlane" name="floor" pos="0 0 0" size="10 10 0.125" type="plane"/>-->
        <body name="torso" pos="0 0 1.4">
            <camera name="track" mode="trackcom" pos="0 -4 0" xyaxes="1 0 0 0 0 1"/>
            <joint armature="0" damping="0" limited="false" name="root" pos="0 0 0" stiffness="0" type="free"/>
            <!-- 第一根 -->
            <geom fromto="0 -.07 0 0 .07 0" name="torso1" size="0.07" type="capsule"/>
            <geom name="head" pos="0 0 .19" size=".09" type="sphere" user="258"/>
            
            <geom fromto="-.01 -.06 -.12 -.01 .06 -.12" name="uwaist" size="0.06" type="capsule"/>

            <body name="lwaist" pos="-.01 0 -0.260" quat="1.000 0 -0.002 0">
                <geom fromto="0 -.06 0 0 .06 0" name="lwaist" size="0.06" type="capsule"/>

                <joint armature="0.02" axis="0 0 1" damping="5" name="abdomen_z" pos="0 0 0.065" range="-45 45" stiffness="20" type="hinge"/>
                <joint armature="0.02" axis="0 1 0" damping="5" name="abdomen_y" pos="0 0 0.065" range="-75 30" stiffness="10" type="hinge"/>
                <body name="pelvis" pos="0 0 -0.165" quat="1.000 0 -0.002 0">
                    <joint armature="0.02" axis="1 0 0" damping="5" name="abdomen_x" pos="0 0 0.1" range="-35 35" stiffness="10" type="hinge"/>
                    <geom fromto="-.02 -.07 0 -.02 .07 0" name="butt" size="0.09" type="capsule"/>


                    <body name="right_thigh" pos="0 -0.1 -0.04">
                        <joint armature="0.01" axis="1 0 0" damping="5" name="right_hip_x" pos="0 0 0" range="-25 5" stiffness="10" type="hinge"/>
                        <joint armature="0.01" axis="0 0 1" damping="5" name="right_hip_z" pos="0 0 0" range="-60 35" stiffness="10" type="hinge"/>
                        <joint armature="0.0080" axis="0 1 0" damping="5" name="right_hip_y" pos="0 0 0" range="-110 20" stiffness="20" type="hinge"/>
                        <geom fromto="0 0 0 0 0.01 -.34" name="right_thigh1" size="0.06" type="capsule"/>

                        <body name="right_shin" pos="0 0.01 -0.403">
                            <joint armature="0.0060" axis="0 -1 0" name="right_knee" pos="0 0 .02" range="-160 -2" type="hinge"/>
                            <geom fromto="0 0 0 0 0 -.3" name="right_shin1" size="0.049" type="capsule"/>

                            <body name="right_foot" pos="0 0 -0.45">
                                <geom name="right_foot" pos="0 0 0.1" size="0.075" type="sphere" user="0"/>
                            </body>
                        </body>
                    </body>

                    <body name="left_thigh" pos="0 0.1 -0.04">
                        <joint armature="0.01" axis="-1 0 0" damping="5" name="left_hip_x" pos="0 0 0" range="-25 5" stiffness="10" type="hinge"/>
                        <joint armature="0.01" axis="0 0 -1" damping="5" name="left_hip_z" pos="0 0 0" range="-60 35" stiffness="10" type="hinge"/>
                        <joint armature="0.01" axis="0 1 0" damping="5" name="left_hip_y" pos="0 0 0" range="-110 20" stiffness="20" type="hinge"/>
                        <geom fromto="0 0 0 0 -0.01 -.34" name="left_thigh1" size="0.06" type="capsule"/>

                        <body name="left_shin" pos="0 -0.01 -0.403">
                            <joint armature="0.0060" axis="0 -1 0" name="left_knee" pos="0 0 .02" range="-160 -2" stiffness="1" type="hinge"/>
                            <geom fromto="0 0 0 0 0 -.3" name="left_shin1" size="0.049" type="capsule"/>

                            <body name="left_foot" pos="0 0 -0.45">
                                <geom name="left_foot" type="sphere" size="0.075" pos="0 0 0.1" user="0" />
                            </body>
                        </body>
                    </body>

                </body>
            </body>
            
            <body name="right_upper_arm" pos="0 -0.17 0.06">
                <joint armature="0.0068" axis="2 1 1" name="right_shoulder1" pos="0 0 0" range="-85 60" stiffness="1" type="hinge"/>
                <joint armature="0.0051" axis="0 -1 1" name="right_shoulder2" pos="0 0 0" range="-85 60" stiffness="1" type="hinge"/>
                <geom fromto="0 0 0 .16 -.16 -.16" name="right_uarm1" size="0.04 0.16" type="capsule"/>

                <body name="right_lower_arm" pos=".18 -.18 -.18">
                    <joint armature="0.0028" axis="0 -1 1" name="right_elbow" pos="0 0 0" range="-90 50" stiffness="0" type="hinge"/>
                    <geom fromto="0.01 0.01 0.01 .17 .17 .17" name="right_larm" size="0.031" type="capsule"/>
                    <geom name="right_hand" pos=".18 .18 .18" size="0.01" type="sphere"/>
                    <camera pos="0 0 0"/>
                </body>
            </body>

            <body name="left_upper_arm" pos="0 0.17 0.06">
                <joint armature="0.0068" axis="2 -1 1" name="left_shoulder1" pos="0 0 0" range="-60 85" stiffness="1" type="hinge"/>
                <joint armature="0.0051" axis="0 1 1" name="left_shoulder2" pos="0 0 0" range="-60 85" stiffness="1" type="hinge"/>
                <geom fromto="0 0 0 .16 .16 -.16" name="left_uarm1" size="0.04 0.16" type="capsule"/>
                <body name="left_lower_arm" pos=".18 .18 -.18">
                    <joint armature="0.0028" axis="0 -1 -1" name="left_elbow" pos="0 0 0" range="-90 50" stiffness="0" type="hinge"/>
                    <geom fromto="0.01 -0.01 0.01 .17 -.17 .17" name="left_larm" size="0.031" type="capsule"/>
                    <geom name="left_hand" pos=".18 -.18 .18" size="0.04" type="sphere"/>
                </body>
            </body>
        </body>
    </worldbody>
    <tendon>
        <fixed name="left_hipknee">
            <joint coef="-1" joint="left_hip_y"/>
            <joint coef="1" joint="left_knee"/>
        </fixed>
        <fixed name="right_hipknee">
            <joint coef="-1" joint="right_hip_y"/>
            <joint coef="1" joint="right_knee"/>
        </fixed>
    </tendon>

    <actuator>
        <motor gear="100" joint="abdomen_y" name="abdomen_y"/>
        <motor gear="100" joint="abdomen_z" name="abdomen_z"/>
        <motor gear="100" joint="abdomen_x" name="abdomen_x"/>
        <motor gear="100" joint="right_hip_x" name="right_hip_x"/>
        <motor gear="100" joint="right_hip_z" name="right_hip_z"/>
        <motor gear="300" joint="right_hip_y" name="right_hip_y"/>
        <motor gear="200" joint="right_knee" name="right_knee"/>
        <motor gear="100" joint="left_hip_x" name="left_hip_x"/>
        <motor gear="100" joint="left_hip_z" name="left_hip_z"/>
        <motor gear="300" joint="left_hip_y" name="left_hip_y"/>
        <motor gear="200" joint="left_knee" name="left_knee"/>
        <motor gear="25" joint="right_shoulder1" name="right_shoulder1"/>
        <motor gear="25" joint="right_shoulder2" name="right_shoulder2"/>
        <motor gear="25" joint="right_elbow" name="right_elbow"/>
        <motor gear="25" joint="left_shoulder1" name="left_shoulder1"/>
        <motor gear="25" joint="left_shoulder2" name="left_shoulder2"/>
        <motor gear="25" joint="left_elbow" name="left_elbow"/>
    </actuator>
</mujoco>

  '''
  for i in range(num_param):
    param_xml = param_xml.replace("{"+str(i+1)+"}", str(parameter[i]))

  fp = open(os.path.join(os.path.dirname(__file__), "assets", filename), "w")
  fp.write(param_xml)
  fp.close()

