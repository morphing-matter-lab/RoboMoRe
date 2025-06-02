rewardfunc_prompts = """
You are a reward engineer trying to write reward functions to solve reinforcement learning tasks as effectively as possible.
Your goal is to write a reward function for the enviroment that will help the agent learn the task described in text.
Task Description: The HalfCheetah is a 2-dimensional robot consisting of 9 body parts and 8 joints connecting them (including two paws). The goal is to apply torque to the joints to make the cheetah run forward (right) as fast as possible, with a positive reward based on the distance moved forward and a negative reward for moving backward.
The cheetah's torso and head are fixed, and torque can only be applied to the other 6 joints over the front and back thighs (which connect to the torso), the shins (which connect to the thighs), and the feet (which connect to the shins), and the goal is to move as fast as possible towards the right by applying torque to the rotors and using fluid friction and you should write a reward function to make the robot move as faster as possible.
Here is the environment codes:
class CheetahEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self,
        xml_file: str = "half_cheetah.xml",
        frame_skip: int = 5,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 0.1,
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

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, action)
        info = {"x_position": x_position_after, "x_velocity": x_velocity, **reward_info}

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, False, False, info

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

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


rewardfunc_format = """
a template can be:
    def _get_rew(self, x_velocity: float, action):
        forward_reward = self._forward_reward_weight * x_velocity
        ctrl_cost = self.control_cost(action)

        reward = forward_reward - ctrl_cost

        reward_info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
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
Description: The HalfCheetah is a 2-dimensional robot consisting of 9 body parts and 8 joints connecting them (including two paws). The goal is to apply torque to the joints to make the cheetah run forward (right) as fast as possible, with a positive reward based on the distance moved forward and a negative reward for moving backward.
The cheetah's torso and head are fixed, and torque can only be applied to the other 6 joints over the front and back thighs (which connect to the torso), the shins (which connect to the thighs), and the feet (which connect to the shins).

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

"""

morphology_div_prompts = """
Please propose a new morphology design, which can promote high fitness function and is quite different from all previous morphology design parameters and design style. 
"""

morphology_format = """
Pleas output in json format without any notes:
{
  "parameters": [<param1>, <param2>, ..., <param24>],
  "desciption": "<your simple design style decription>",
}
Parameters Description:
# param1: start point of torso (x), negtive
# param2: end point of torso (x) and start point of head (x), positive
# param3: end point of head (x).
# param4: end point of head (z).
# back leg parameters
# param5: end point of bthigh (x) and start point of bshin (x), sometimes positive.
# param6: end point of bthigh (z) and start point of bshin (z), negative.
# param7: end point of bshin (x) and start point of bfoot (x), sometimes negtive.
# param8: end point of bshin (z) and start point of bfoot (z), negative.
# param9: end point of bfoot (x), sometimes positive.
# param10: end point of bfoot (z), negative.
# forward leg parameters
# param11: end point of fthigh (x) and start point of fshin (x), sometimes negative.
# param12: end point of fthigh (z) and start point of fshin (z), negative.
# param13: end point of fshin (x) and start point of ffoot (x), sometimes positive.
# param14: end point of fshin (z) and start point of ffoot (z), negative.
# param15: end point of ffoot (x), sometimes positive.
# param16: end point of ffoot (z), negative.
# size information
# param17: torso capsule size.
# param18: head capsule size.
# param19: bthigh capsule size.
# param20: bshin capsule size.
# param21: bfoot capsule size.
# param22: fthigh capsule size.
# param23: fshin capsule size.
# param24: ffoot capsule size.

Note: 
1. Please ensure the number of params is accurate.
2. You must consider structure of robot carefully and give accurate parameters step by step.
3. For reducing material cost to ensure the effieciency of robot design, you should reduce redundant paramters (e.g., smaller geom size) and increase paramters who control the robot (e.g., longer lengths).
4. Your design should fit the control gear and others parts of robots well.
"""


morphology_improve_prompts="""
Role: You are a robot designer trying to design robot parameters to increase the fitness function as effective as possible.
Task: Your task is to design parameters of robot that will help agent achieve the fitness function as high as possible.
fintess function: walk distance/material cost.
Description: The HalfCheetah is a 2-dimensional robot consisting of 9 body parts and 8 joints connecting them (including two paws). The goal is to apply torque to the joints to make the cheetah run forward (right) as fast as possible, with a positive reward based on the distance moved forward and a negative reward for moving backward.
The cheetah's torso and head are fixed, and torque can only be applied to the other 6 joints over the front and back thighs (which connect to the torso), the shins (which connect to the thighs), and the feet (which connect to the shins).

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

There are also some design parameters and their fitness. 
Please carefully observe these parameters and their fitness, try to design a new parameter to further improve the fitness.

"""

reward_improve_prompts = """
You are a reward engineer trying to write reward functions to solve reinforcement learning tasks as effectively as possible.
Your goal is to write a reward function for the enviroment that will help the agent learn the task described in text.
Task Description: The HalfCheetah is a 2-dimensional robot consisting of 9 body parts and 8 joints connecting them (including two paws). The goal is to apply torque to the joints to make the cheetah run forward (right) as fast as possible, with a positive reward based on the distance moved forward and a negative reward for moving backward.
The cheetah's torso and head are fixed, and torque can only be applied to the other 6 joints over the front and back thighs (which connect to the torso), the shins (which connect to the thighs), and the feet (which connect to the shins), and the goal is to move as fast as possible towards the right by applying torque to the rotors and using fluid friction and you should write a reward function to make the robot move as faster as possible.
Here is the environment codes:
class CheetahEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self,
        xml_file: str = "half_cheetah.xml",
        frame_skip: int = 5,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1.0,
        ctrl_cost_weight: float = 0.1,
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

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        x_position_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, action)
        info = {"x_position": x_position_after, "x_velocity": x_velocity, **reward_info}

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, False, False, info

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

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
        }  

There are also some reward functions and their fitness. 
You should observe the highest rewards from the fitness function and further encourage those motion patterns. Also, analyze the low fitness rewards, identify their shortcomings, and avoid those motion patterns. This is very important.Please carefully observe these reward funcions and their fitness, try to write a reward function to further improve the fitness.

"""