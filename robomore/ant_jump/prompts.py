rewardfunc_prompts = """
You are a reward engineer trying to write reward functions to solve reinforcement learning tasks as effectively as possible.
Your goal is to write a reward function for the enviroment that will help the agent learn the task described in text.
Task Description: The ant is a 3D quadruped robot consisting of a torso (free rotational body) with four legs attached to it, where each leg has two body parts. 
The goal is to coordinate the four legs to jump in the up direction by applying torque to the eight hinges connecting the two body parts of each leg and the torso (nine body parts and eight hinges), You should write a reward function to make the robot jump as higher as possible.
Here is the environment codes:

Here is the environment codes:
class AntEnv(MujocoEnv, utils.EzPickle):
    
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self,
        xml_file: str = "ant.xml",
        frame_skip: int = 5,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1,
        ctrl_cost_weight: float = 0.5,
        contact_cost_weight: float = 5e-4,
        healthy_reward: float = 1.0,
        main_body: Union[int, str] = 1,
        terminate_when_unhealthy: bool = True,
        healthy_z_range: Tuple[float, float] = (0.2, 1.8),
        contact_force_range: Tuple[float, float] = (-1.0, 1.0),
        reset_noise_scale: float = 0.1,
        exclude_current_positions_from_observation: bool = True,
        include_cfrc_ext_in_observation: bool = True,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            forward_reward_weight,
            ctrl_cost_weight,
            contact_cost_weight,
            healthy_reward,
            main_body,
            terminate_when_unhealthy,
            healthy_z_range,
            contact_force_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            include_cfrc_ext_in_observation,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._contact_force_range = contact_force_range

        self._main_body = main_body

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        self._include_cfrc_ext_in_observation = include_cfrc_ext_in_observation

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,  # needs to be defined after
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

        obs_size = self.data.qpos.size + self.data.qvel.size
        obs_size -= 2 * exclude_current_positions_from_observation
        obs_size += self.data.cfrc_ext[1:].size * include_cfrc_ext_in_observation

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = {
            "skipped_qpos": 2 * exclude_current_positions_from_observation,
            "qpos": self.data.qpos.size
            - 2 * exclude_current_positions_from_observation,
            "qvel": self.data.qvel.size,
            "cfrc_ext": self.data.cfrc_ext[1:].size * include_cfrc_ext_in_observation,
        }

    @property
    def healthy_reward(self):
        return self.is_healthy * self._healthy_reward

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def contact_forces(self):
        raw_contact_forces = self.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces)
        )
        return contact_cost

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z
        return is_healthy

    def step(self, action):
        xy_position_before = self.data.body(self._main_body).xpos[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.data.body(self._main_body).xpos[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, action)
        terminated = (not self.is_healthy) and self._terminate_when_unhealthy
        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        if self._include_cfrc_ext_in_observation:
            contact_force = self.contact_forces[1:].flatten()
            return np.concatenate((position, velocity, contact_force))
        else:
            return np.concatenate((position, velocity))

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
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
Please write a new reward function to encourage more robot motion behaviours, which can promote high fitness function and is quite different from all previous reward functions in the design style. 
"""




morphology_prompts = """
Role: You are a robot designer trying to design robot parameters to increase the fitness function as effective as possible.
Task: Your task is to design parameters of robot that will help agent achieve the fitness function as high as possible.
fintess function: jump height/material cost.
Description: The ant is a 3D quadruped robot consisting of a torso (free rotational body) with four legs attached to it, where each leg has two body parts. The goal is to coordinate the four legs to jump in the up direction by applying torque to the eight hinges connecting the two body parts of each leg and the torso (nine body parts and eight hinges).

Here is the xml file:
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
      <geom name="torso_geom" pos="0 0 0" size="{param1}" type="sphere"/>
      <joint armature="0" damping="0" limited="false" margin="0.01" name="root" pos="0 0 0" type="free"/>
      <body name="front_left_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 {param2} {param3} 0.0" name="aux_1_geom" size="{param8}" type="capsule"/>
        <body name="aux_1" pos="{param2} {param3} 0">
          <joint axis="0 0 1" name="hip_1" pos="0.0 0.0 0.0" range="-40 40" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 {param4} {param5} 0.0" name="left_leg_geom" size="{param9}" type="capsule" />
          <body pos="{param4} {param5} 0" >
            <joint axis="-1 1 0" name="ankle_1" pos="0.0 0.0 0.0" range="30 100" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 {param6} {param7} 0.0" name="left_ankle_geom" size="{param10}" type="capsule"/>
          </body>
        </body>
      </body>
      <body name="front_right_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 -{param2} {param3} 0.0" name="aux_2_geom" size="{param8}" type="capsule"/>
        <body name="aux_2" pos="-{param2} {param3} 0">
          <joint axis="0 0 1" name="hip_2" pos="0.0 0.0 0.0" range="-40 40" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 -{param4} {param5} 0.0" name="right_leg_geom" size="{param9}" type="capsule"/>
          <body pos="-{param4} {param5} 0" >
            <joint axis="1 1 0" name="ankle_2" pos="0.0 0.0 0.0" range="-100 -30" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 -{param6} {param7} 0.0" name="right_ankle_geom" size="{param10}" type="capsule"/>
          </body>
        </body>
      </body>
      <body name="left_back_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 -{param2} -{param3} 0.0" name="aux_3_geom" size="{param8}" type="capsule"/>
        <body name="aux_3" pos="-{param2} -{param3} 0">
          <joint axis="0 0 1" name="hip_3" pos="0.0 0.0 0.0" range="-40 40" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 -{param4} -{param5} 0.0" name="back_leg_geom" size="{param9}" type="capsule"/>
          <body pos="-{param4} {param5} 0" >
            <joint axis="-1 1 0" name="ankle_3" pos="0.0 0.0 0.0" range="-100 -30" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 -{param6} -{param7} 0.0" name="third_ankle_geom" size="{param10}" type="capsule"/>
          </body>
        </body>
      </body>
      <body name="right_back_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 {param2} -{param3} 0.0" name="aux_4_geom" size="{param8}" type="capsule"/>
        <body name="aux_4" pos=" {param2} -{param3} 0">
          <joint axis="0 0 1" name="hip_4" pos="0.0 0.0 0.0" range="-40 40" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 {param4} -{param5} 0.0" name="rightback_leg_geom" size="{param9}" type="capsule"/>
          <body pos="{param4} -{param5} 0" >
            <joint axis="1 1 0" name="ankle_4" pos="0.0 0.0 0.0" range="30 100" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 {param6} -{param7} 0.0" name="fourth_ankle_geom" size="{param10}" type="capsule"/>
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
1. For reducing material cost to ensure the effieciency of robot design, you should reduce redundant paramters (e.g., smaller geom size) and increase paramters who control the robot (e.g., longer leags, ankles).
2. Your design should fit the control gear and others parts of robots well.


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
Note: Please ensure the number of params is accurate, and all params are positive.
"""


morphology_improve_prompts="""
Role: You are a robot designer trying to design robot parameters to increase the fitness function as effective as possible.
Task: Your task is to design parameters of robot that will help agent achieve the fitness function as high as possible.
fintess function: jump height/material cost.
Description: The ant is a 3D quadruped robot consisting of a torso (free rotational body) with four legs attached to it, where each leg has two body parts. The goal is to coordinate the four legs to jump in the up direction by applying torque to the eight hinges connecting the two body parts of each leg and the torso (nine body parts and eight hinges).

Here is the xml file:
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
      <geom name="torso_geom" pos="0 0 0" size="{param1}" type="sphere"/>
      <joint armature="0" damping="0" limited="false" margin="0.01" name="root" pos="0 0 0" type="free"/>
      <body name="front_left_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 {param2} {param3} 0.0" name="aux_1_geom" size="{param8}" type="capsule"/>
        <body name="aux_1" pos="{param2} {param3} 0">
          <joint axis="0 0 1" name="hip_1" pos="0.0 0.0 0.0" range="-40 40" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 {param4} {param5} 0.0" name="left_leg_geom" size="{param9}" type="capsule" />
          <body pos="{param4} {param5} 0" >
            <joint axis="-1 1 0" name="ankle_1" pos="0.0 0.0 0.0" range="30 100" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 {param6} {param7} 0.0" name="left_ankle_geom" size="{param10}" type="capsule"/>
          </body>
        </body>
      </body>
      <body name="front_right_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 -{param2} {param3} 0.0" name="aux_2_geom" size="{param8}" type="capsule"/>
        <body name="aux_2" pos="-{param2} {param3} 0">
          <joint axis="0 0 1" name="hip_2" pos="0.0 0.0 0.0" range="-40 40" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 -{param4} {param5} 0.0" name="right_leg_geom" size="{param9}" type="capsule"/>
          <body pos="-{param4} {param5} 0" >
            <joint axis="1 1 0" name="ankle_2" pos="0.0 0.0 0.0" range="-100 -30" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 -{param6} {param7} 0.0" name="right_ankle_geom" size="{param10}" type="capsule"/>
          </body>
        </body>
      </body>
      <body name="left_back_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 -{param2} -{param3} 0.0" name="aux_3_geom" size="{param8}" type="capsule"/>
        <body name="aux_3" pos="-{param2} -{param3} 0">
          <joint axis="0 0 1" name="hip_3" pos="0.0 0.0 0.0" range="-40 40" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 -{param4} -{param5} 0.0" name="back_leg_geom" size="{param9}" type="capsule"/>
          <body pos="-{param4} {param5} 0" >
            <joint axis="-1 1 0" name="ankle_3" pos="0.0 0.0 0.0" range="-100 -30" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 -{param6} -{param7} 0.0" name="third_ankle_geom" size="{param10}" type="capsule"/>
          </body>
        </body>
      </body>
      <body name="right_back_leg" pos="0 0 0">
        <geom fromto="0.0 0.0 0.0 {param2} -{param3} 0.0" name="aux_4_geom" size="{param8}" type="capsule"/>
        <body name="aux_4" pos=" {param2} -{param3} 0">
          <joint axis="0 0 1" name="hip_4" pos="0.0 0.0 0.0" range="-40 40" type="hinge"/>
          <geom fromto="0.0 0.0 0.0 {param4} -{param5} 0.0" name="rightback_leg_geom" size="{param9}" type="capsule"/>
          <body pos="{param4} -{param5} 0" >
            <joint axis="1 1 0" name="ankle_4" pos="0.0 0.0 0.0" range="30 100" type="hinge"/>
            <geom fromto="0.0 0.0 0.0 {param6} -{param7} 0.0" name="fourth_ankle_geom" size="{param10}" type="capsule"/>
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
1. For reducing material cost to ensure the effieciency of robot design, you should reduce redundant paramters (e.g., smaller geom size) and increase paramters who control the robot (e.g., longer leags, ankles).
2. Your design should fit the control gear and others parts of robots well.

There are also some design parameters and their fitness. 
Please carefully observe these parameters and their fitness, try to design a new parameter to further improve the fitness.

"""

reward_improve_prompts = """

You are a reward engineer trying to write reward functions to solve reinforcement learning tasks as effectively as possible.
Your goal is to write a reward function for the enviroment that will help the agent learn the task described in text.

Task Description: The ant is a 3D quadruped robot consisting of a torso (free rotational body) with four legs attached to it, where each leg has two body parts. 
The goal is to coordinate the four legs to jump in the up direction by applying torque to the eight hinges connecting the two body parts of each leg and the torso (nine body parts and eight hinges), You should write a reward function to make the robot jump as higher as possible.
Here is the environment codes:
Here is the environment codes:
class AntEnv(MujocoEnv, utils.EzPickle):
    
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self,
        xml_file: str = "ant.xml",
        frame_skip: int = 5,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1,
        ctrl_cost_weight: float = 0.5,
        contact_cost_weight: float = 5e-4,
        healthy_reward: float = 1.0,
        main_body: Union[int, str] = 1,
        terminate_when_unhealthy: bool = True,
        healthy_z_range: Tuple[float, float] = (0.2, 1.8),
        contact_force_range: Tuple[float, float] = (-1.0, 1.0),
        reset_noise_scale: float = 0.1,
        exclude_current_positions_from_observation: bool = True,
        include_cfrc_ext_in_observation: bool = True,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            forward_reward_weight,
            ctrl_cost_weight,
            contact_cost_weight,
            healthy_reward,
            main_body,
            terminate_when_unhealthy,
            healthy_z_range,
            contact_force_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            include_cfrc_ext_in_observation,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._contact_force_range = contact_force_range

        self._main_body = main_body

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        self._include_cfrc_ext_in_observation = include_cfrc_ext_in_observation

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,  # needs to be defined after
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

        obs_size = self.data.qpos.size + self.data.qvel.size
        obs_size -= 2 * exclude_current_positions_from_observation
        obs_size += self.data.cfrc_ext[1:].size * include_cfrc_ext_in_observation

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = {
            "skipped_qpos": 2 * exclude_current_positions_from_observation,
            "qpos": self.data.qpos.size
            - 2 * exclude_current_positions_from_observation,
            "qvel": self.data.qvel.size,
            "cfrc_ext": self.data.cfrc_ext[1:].size * include_cfrc_ext_in_observation,
        }

    @property
    def healthy_reward(self):
        return self.is_healthy * self._healthy_reward

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def contact_forces(self):
        raw_contact_forces = self.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces)
        )
        return contact_cost

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z
        return is_healthy

    def step(self, action):
        xy_position_before = self.data.body(self._main_body).xpos[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.data.body(self._main_body).xpos[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, action)
        terminated = (not self.is_healthy) and self._terminate_when_unhealthy
        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        if self._include_cfrc_ext_in_observation:
            contact_force = self.contact_forces[1:].flatten()
            return np.concatenate((position, velocity, contact_force))
        else:
            return np.concatenate((position, velocity))

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
        }

There are also some reward functions and their fitness. 
Please carefully observe these reward funcions and their fitness, try to write a reward function to further improve the fitness.

"""