rewardfunc_prompts = """
You are a reward engineer trying to write reward functions to solve reinforcement learning tasks as effectively as possible.
Your goal is to write a reward function for the enviroment that will help the agent learn the task described in text.

Task Description: The hopper is a two-dimensional one-legged figure consisting of four main body parts - the torso at the top, the thigh in the middle, the leg at the bottom, and a single foot on which the entire body rests.
The goal is to make hops that move in the forward (right) direction by applying torque to the three hinges that connect the four body parts, you should write a reward function to make the robot move as faster as possible.

Here is the environment codes:
class HopperEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self,
        xml_file: str = "hopper.xml",
        frame_skip: int = 4,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 1e-3,
        healthy_reward: float = 1.0,
        terminate_when_unhealthy: bool = True,
        healthy_state_range: Tuple[float, float] = (-100.0, 100.0),
        healthy_z_range: Tuple[float, float] = (0.7, float("inf")),
        healthy_angle_range: Tuple[float, float] = (-0.2, 0.2),
        reset_noise_scale: float = 5e-3,
        exclude_current_positions_from_observation: bool = True,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_state_range,
            healthy_z_range,
            healthy_angle_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy

        self._healthy_state_range = healthy_state_range
        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = (
            self.data.qpos.size
            + self.data.qvel.size
            - exclude_current_positions_from_observation
        )
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = {
            "skipped_qpos": 1 * exclude_current_positions_from_observation,
            "qpos": self.data.qpos.size
            - 1 * exclude_current_positions_from_observation,
            "qvel": self.data.qvel.size,
        }

    @property
    def healthy_reward(self):
        return self.is_healthy * self._healthy_reward

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def is_healthy(self):
        z, angle = self.data.qpos[1:3]
        state = self.state_vector()[2:]

        min_state, max_state = self._healthy_state_range
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_state = np.all(np.logical_and(min_state < state, state < max_state))
        healthy_z = min_z < z < max_z
        healthy_angle = min_angle < angle < max_angle

        is_healthy = all((healthy_state, healthy_z, healthy_angle))

        return is_healthy

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = np.clip(self.data.qvel.flatten(), -10, 10)

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, action)
        terminated = (not self.is_healthy) and self._terminate_when_unhealthy
        info = {
            "x_position": x_position_after,
            "z_distance_from_origin": self.data.qpos[1] - self.init_qpos[1],
            "x_velocity": x_velocity,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "z_distance_from_origin": self.data.qpos[1] - self.init_qpos[1],
        }

    
"""


zeroshot_rewardfunc_format = """
a template reward can be:
    def _get_rew(self, x_velocity: float, action):
        <reward function code you should write>
        return reward, reward_info    

The output of the reward function should consist of two items:  
(1) `reward`, which is the total reward.  
(2) `reward_info`, a dictionary of each individual reward component.  

The code output should be formatted as a Python code string: `'```python ...```'`.  

Some helpful tips for writing the reward function code:  
(1) You may find it helpful to normalize the reward to a fixed range by applying transformations like `numpy.exp` to the overall reward or its components.    
(3) Make sure the type of each input variable is correctly specified and the function name is "def _get_rew():"
(4) Most importantly, the reward code's input variables must contain only attributes of the provided environment class definition (namely, variables that have the prefix `self.`). Under no circumstances can you introduce new input variables.
"""


rewardfunc_div_prompts = """
Please write a new reward function to encourage different and efficient motion behaviors. Please choose a different motion behavior and ensure the behavior maximizes task fitness through this specific motion style.
"""

# rewardfunc_div_prompts = """
# Please write a new reward function to encourage more robot motion behaviours, which can promote high fitness function and is quite different from all previous reward functions in the design style. 
# """


rewardfunc_format = """
a template can be:
    def _get_rew(self, x_velocity: float, action):
        forward_reward = self._forward_reward_weight * x_velocity
        healthy_reward = self.healthy_reward
        rewards = forward_reward + healthy_reward

        ctrl_cost = self.control_cost(action)
        costs = ctrl_cost

        reward = rewards - costs

        reward_info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
        }

        return reward, reward_info

The output of the reward function should consist of two items:  
(1) `reward`, which is the total reward.  
(2) `reward_info`, a dictionary of each individual reward component.  

The code output should be formatted as a Python code string: `'```python ...```'`.  

Some helpful tips for writing the reward function code:  
(1) You may find it helpful to normalize the reward to a fixed range by applying transformations like `numpy.exp` to the overall reward or its components.    
(3) Make sure the type of each input variable is correctly specified and the function name is "def _get_rew():"
(4) Most importantly, the reward code's input variables must contain only attributes of the provided environment class definition (namely, variables that have the prefix `self.`). Under no circumstances can you introduce new input variables.
"""


morphology_prompts = """
Role: You are a robot designer trying to design robot parameters to increase the fitness function as effective as possible.
Task: Your task is to design parameters of robot that will help agent achieve the fitness function as high as possible.
fintess function: walk distance/material cost.
Description: The hopper is a two-dimensional one-legged figure consisting of four main body parts - the torso at the top, the thigh in the middle, the leg at the bottom, and a single foot on which the entire body rests.
The goal is to make hops that move in the forward (right) direction by applying torque to the three hinges that connect the four body parts, you should write a reward function to make the robot move as faster as possible.

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
        <geom fromto="0 0 {param1} 0 0 {param2}" name="torso_geom" size="{param8}" type="capsule" friction="0.9"/>
        <body name="thigh">
          <joint axis="0 -1 0" name="thigh_joint" pos="0 0 {param2}" range="-150 0" type="hinge"/>
          <geom fromto="0 0 {param2} 0 0 {param3}" name="thigh_geom" size="{param8}" type="capsule" friction="0.9"/>
          <body name="leg">
            <joint axis="0 -1 0" name="leg_joint" pos="0 0 {param3}" range="-150 0" type="hinge"/>
            <geom fromto="0 0 {param3} 0 0 {param4}" name="leg_geom" size="{param9}" type="capsule" friction="0.9"/>
            <body name="foot">
              <joint axis="0 -1 0" name="foot_joint" pos="0 0 {4}" range="-45 45" type="hinge"/>
              <geom fromto="{param5} 0 {param4} {param6} 0 {param4}" name="foot_geom" size="{param10}" type="capsule" friction="2.0"/>
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

"""

morphology_div_prompts = """
Please propose a new morphology design, which can promote high fitness function and is quite different from all previous morphology designs in the design style. 
"""

morphology_format = """
Pleas output in json format without any notes:
{
  "parameters": [<param1>, <param2>, ..., <param10>],
  "desciption": "<your simple design style decription>",
}

Parameters Description:
#   param1 is the upper attachment point of torso, should be hightest, positive.
#   param2 is the lower attachment point of torso and upper attachment point of thigh, should less than param1, positive.
#   param3 is the lower attachment point of thigh and upper attachment point of leg, should less than param2, positive
#   param4 is the lower attachment point of leg and upper attachment point of foot, should less than param3, positive.
#   param5 is the upper attachment point of foot, must be positive.
#   param6 is the lower attachment point of foot, must be negative. Foot should be attatched to the leg.
#   param7 is size of torso.
#   param8 is size of thigh.
#   param9 is size of leg.
#   param10 is size of foot.

Note: 
1. Please ensure the number of params is accurate 
2. You must consider structure of robot carefully and give accurate parameters step by step.
3. For reducing material cost to ensure the effieciency of robot design, you should reduce redundant paramters (e.g., smaller geom size) and increase paramters who control the robot (e.g., longer legs).
4. Your design should fit the control gear and others parts of robots well.

"""


morphology_improve_prompts="""
Role: You are a robot designer trying to design robot parameters to increase the fitness function as effective as possible.
Task: Your task is to design parameters of robot that will help agent achieve the fitness function as high as possible.
fintess function: walk distance/material cost.
Description: The hopper is a two-dimensional one-legged figure consisting of four main body parts - the torso at the top, the thigh in the middle, the leg at the bottom, and a single foot on which the entire body rests.
The goal is to make hops that move in the forward (right) direction by applying torque to the three hinges that connect the four body parts, you should write a reward function to make the robot move as faster as possible.

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
        <geom fromto="0 0 {param1} 0 0 {param2}" name="torso_geom" size="{param8}" type="capsule" friction="0.9"/>
        <body name="thigh">
          <joint axis="0 -1 0" name="thigh_joint" pos="0 0 {param2}" range="-150 0" type="hinge"/>
          <geom fromto="0 0 {param2} 0 0 {param3}" name="thigh_geom" size="{param8}" type="capsule" friction="0.9"/>
          <body name="leg">
            <joint axis="0 -1 0" name="leg_joint" pos="0 0 {param3}" range="-150 0" type="hinge"/>
            <geom fromto="0 0 {param3} 0 0 {param4}" name="leg_geom" size="{param9}" type="capsule" friction="0.9"/>
            <body name="foot">
              <joint axis="0 -1 0" name="foot_joint" pos="0 0 {4}" range="-45 45" type="hinge"/>
              <geom fromto="{param5} 0 {param4} {param6} 0 {param4}" name="foot_geom" size="{param10}" type="capsule" friction="2.0"/>
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

1. For reducing material cost to ensure the effieciency of robot design, you should reduce redundant paramters (e.g., smaller geom size) and increase paramters who control the robot (e.g., longer leags, ankles).
2. Your design should fit the control gear and others parts of robots well.

There are also some design parameters and their fitness. Please carefully observe these parameters and their fitness, try to design a new parameter to further improve the fitness.

Attention:
1.You must observe the parameters from the highest fitness function and further encourage this design. Also, analyze the low fitness parameters, identify their shortcomings, and avoid those design. 
2.You must carefully observe these parameters and their fitness, try to design a new parameter which is different with the best parameter to further improve the fitness. This is very important.

"""

reward_improve_prompts = """
You are a reward engineer trying to write reward functions to solve reinforcement learning tasks as effectively as possible.
Your goal is to write a reward function for the enviroment that will help the agent learn the task described in text.

Task Description: The hopper is a two-dimensional one-legged figure consisting of four main body parts - the torso at the top, the thigh in the middle, the leg at the bottom, and a single foot on which the entire body rests.
The goal is to make hops that move in the forward (right) direction by applying torque to the three hinges that connect the four body parts, you should write a reward function to make the robot move as faster as possible.

Here is the environment codes:
class HopperEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self,
        xml_file: str = "hopper.xml",
        frame_skip: int = 4,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 1e-3,
        healthy_reward: float = 1.0,
        terminate_when_unhealthy: bool = True,
        healthy_state_range: Tuple[float, float] = (-100.0, 100.0),
        healthy_z_range: Tuple[float, float] = (0.7, float("inf")),
        healthy_angle_range: Tuple[float, float] = (-0.2, 0.2),
        reset_noise_scale: float = 5e-3,
        exclude_current_positions_from_observation: bool = True,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_state_range,
            healthy_z_range,
            healthy_angle_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy

        self._healthy_state_range = healthy_state_range
        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = (
            self.data.qpos.size
            + self.data.qvel.size
            - exclude_current_positions_from_observation
        )
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = {
            "skipped_qpos": 1 * exclude_current_positions_from_observation,
            "qpos": self.data.qpos.size
            - 1 * exclude_current_positions_from_observation,
            "qvel": self.data.qvel.size,
        }

    @property
    def healthy_reward(self):
        return self.is_healthy * self._healthy_reward

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def is_healthy(self):
        z, angle = self.data.qpos[1:3]
        state = self.state_vector()[2:]

        min_state, max_state = self._healthy_state_range
        min_z, max_z = self._healthy_z_range
        min_angle, max_angle = self._healthy_angle_range

        healthy_state = np.all(np.logical_and(min_state < state, state < max_state))
        healthy_z = min_z < z < max_z
        healthy_angle = min_angle < angle < max_angle

        is_healthy = all((healthy_state, healthy_z, healthy_angle))

        return is_healthy

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = np.clip(self.data.qvel.flatten(), -10, 10)

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, action)
        terminated = (not self.is_healthy) and self._terminate_when_unhealthy
        info = {
            "x_position": x_position_after,
            "z_distance_from_origin": self.data.qpos[1] - self.init_qpos[1],
            "x_velocity": x_velocity,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "z_distance_from_origin": self.data.qpos[1] - self.init_qpos[1],
        }
        
        
        
There are also some reward functions and their fitness. 
1.You should observe the highest rewards from the fitness function and further encourage those motion patterns. Also, analyze the low fitness rewards, identify their shortcomings, and avoid those motion patterns. This is very important.
2.Please carefully observe these reward funcions and their fitness, try to write a reward function which is different with these reward functions to further improve the fitness.

"""