rewardfunc_prompts = """
You are a reward engineer trying to write reward functions to solve reinforcement learning tasks as effectively as possible.
Your goal is to write a reward function for the enviroment that will help the agent learn the task described in text.

Task Description: The swimmers consist of three or more segments ('***links***') and one less articulation joints ('***rotors***') - one rotor joint connects exactly two links to form a linear chain.
The swimmer is suspended in a two-dimensional pool and always starts in the same position (subject to some deviation drawn from a uniform distribution),
and the goal is to move as fast as possible towards the right by applying torque to the rotors and using fluid friction and you should write a reward function to make the robot move as faster as possible.

Here is the environment codes:
class SwimmerEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self,
        xml_file: str = "swimmer.xml",
        frame_skip: int = 4,
        default_camera_config: Dict[str, Union[float, int]] = {},
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 1e-4,
        reset_noise_scale: float = 0.1,
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
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight

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
            - 2 * exclude_current_positions_from_observation
        )
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = {
            "skipped_qpos": 2 * exclude_current_positions_from_observation,
            "qpos": self.data.qpos.size
            - 2 * exclude_current_positions_from_observation,
            "qvel": self.data.qvel.size,
        }

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        xy_position_before = self.data.qpos[0:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.data.qpos[0:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, action)
        info = {
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, False, False, info

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        observation = np.concatenate([position, velocity]).ravel()
        return observation

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
Description: The swimmers consist of three or more segments ('***links***') and one less articulation joints ('***rotors***') - one rotor joint connects exactly two links to form a linear chain.
The swimmer is suspended in a two-dimensional pool and always starts in the same position (subject to some deviation drawn from a uniform distribution),
and the goal is to move as fast as possible towards the right by applying torque to the rotors and using fluid friction and you should write a reward function to make the robot move as faster as possible.

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
      <geom density="1000" fromto="{param1} 0 0 0 0 0" size="{param4}" type="capsule"/>
      <joint axis="1 0 0" name="slider1" pos="0 0 0" type="slide"/>
      <joint axis="0 1 0" name="slider2" pos="0 0 0" type="slide"/>
      <joint axis="0 0 1" name="free_body_rot" pos="0 0 0" type="hinge"/>
      <body name="mid" pos="0 0 0">
        <geom density="1000" fromto="0 0 0 -{param2} 0 0" size="{param5}" type="capsule"/>
        <joint axis="0 0 1" limited="true" name="motor1_rot" pos="0 0 0" range="-100 100" type="hinge"/>
        <body name="back" pos="-{param2} 0 0">
          <geom density="1000" fromto="0 0 0 -{param3} 0 0" size="{param6}" type="capsule"/>
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

"""

morphology_div_prompts = """
Please propose a new morphology design, which can promote high fitness function and is quite different from all previous morphology designs in the design style. 
"""

morphology_format = """
Pleas output in json format without any notes:
{
  "parameters": [<param1>, <param2>, ..., <param6>],
  "desciption": "<your simple design style decription>",
}

Parameters Description:
#   param1 is the length of first segment, positive.
#   param2 is the length of second segment, positive.
#   param3 is the length of third segment, positive.
#   param4 is the size of first segment, positive.
#   param5 is the size of second segment, positive.
#   param6 is the size of third segment, positive.

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
Description: The swimmers consist of three or more segments ('***links***') and one less articulation joints ('***rotors***') - one rotor joint connects exactly two links to form a linear chain.
The swimmer is suspended in a two-dimensional pool and always starts in the same position (subject to some deviation drawn from a uniform distribution),
and the goal is to move as fast as possible towards the right by applying torque to the rotors and using fluid friction and you should write a reward function to make the robot move as faster as possible.

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
      <geom density="1000" fromto="{param1} 0 0 0 0 0" size="{param4}" type="capsule"/>
      <joint axis="1 0 0" name="slider1" pos="0 0 0" type="slide"/>
      <joint axis="0 1 0" name="slider2" pos="0 0 0" type="slide"/>
      <joint axis="0 0 1" name="free_body_rot" pos="0 0 0" type="hinge"/>
      <body name="mid" pos="0 0 0">
        <geom density="1000" fromto="0 0 0 -{param2} 0 0" size="{param5}" type="capsule"/>
        <joint axis="0 0 1" limited="true" name="motor1_rot" pos="0 0 0" range="-100 100" type="hinge"/>
        <body name="back" pos="-{param2} 0 0">
          <geom density="1000" fromto="0 0 0 -{param3} 0 0" size="{param6}" type="capsule"/>
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

1. For reducing material cost to ensure the effieciency of robot design, you should reduce redundant paramters (e.g., smaller geom size) and increase paramters who control the robot (e.g., longer geom length).
2. Your design should fit the control gear and others parts of robots well.

There are also some design parameters and their fitness. 
Please carefully observe these parameters and their fitness, try to design a new parameter to further improve the fitness.

1.You should observe the parameters from the highest fitness function and further encourage this design. Also, analyze the low fitness parameters, identify their shortcomings, and avoid those design. 
2.Please carefully observe these parameters and their fitness, try to design a new parameter which is different with these parameters to further improve the fitness.
3.Please explore some different design parameters accoring to advantages and shortcomings of previous parameters. 
"""

reward_improve_prompts = """
You are a reward engineer trying to write reward functions to solve reinforcement learning tasks as effectively as possible.
Your goal is to write a reward function for the enviroment that will help the agent learn the task described in text.

Task Description: The swimmers consist of three or more segments ('***links***') and one less articulation joints ('***rotors***') - one rotor joint connects exactly two links to form a linear chain.
The swimmer is suspended in a two-dimensional pool and always starts in the same position (subject to some deviation drawn from a uniform distribution),
and the goal is to move as fast as possible towards the right by applying torque to the rotors and using fluid friction and you should write a reward function to make the robot move as faster as possible.

Here is the environment codes:
class SwimmerEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self,
        xml_file: str = "swimmer.xml",
        frame_skip: int = 4,
        default_camera_config: Dict[str, Union[float, int]] = {},
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 1e-4,
        reset_noise_scale: float = 0.1,
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
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight

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
            - 2 * exclude_current_positions_from_observation
        )
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = {
            "skipped_qpos": 2 * exclude_current_positions_from_observation,
            "qpos": self.data.qpos.size
            - 2 * exclude_current_positions_from_observation,
            "qvel": self.data.qvel.size,
        }

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        xy_position_before = self.data.qpos[0:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.data.qpos[0:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, action)
        info = {
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, False, False, info

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        observation = np.concatenate([position, velocity]).ravel()
        return observation

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
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
        }

There are also some reward functions and their fitness. 

1.You should observe the highest rewards from the fitness function and further encourage those motion patterns. Also, analyze the low fitness rewards, identify their shortcomings, and avoid those motion patterns. This is very important.
2.Please carefully observe these reward funcions and their fitness, try to write a reward function which is different with these reward functions to further improve the fitness.
"""