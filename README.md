# Safe Multi-Agent Isaac Gym Benchmark (Safe MAIG)

Safe Multi-Agent Isaac Gym benchmark is for safe multi-agent reinforcement learning research.

<div align=center>
 <img src="https://github.com/chauncygu/Safe-Multi-Agent-Isaac-Gym/blob/main/assets/image_folder/all-demos.gif" width="550"/> 
 </div>
<div align=center>
<center style="color:#000000;text-decoration:underline"> Safe multi-agent Isaac Gym demos. </center>
 </div>

***
The README is organized as follows:

- [About this repository](#about-this-repository)
- [Installation](#installation)
  * [Pre-requisites](#pre-requisites)
  * [install this repo](#install-this-repo)
- [Running the benchmarks](#running-the-benchmarks)
- [Select an algorithm](#select-an-algorithm)
- [Select tasks](#select-tasks)
  * [HandOver Environments](#handover-environments)
    + [<span id="obs1">Observation Space</span>](#-span-id--obs1--observation-space--span-)
    + [<span id="action1">Action Space</span>](#-span-id--action1--action-space--span-)
    + [<span id="r1">Rewards</span>](#-span-id--r1--rewards--span-)
  * [HandCatchUnderarm Environments](#handcatchunderarm-environments)
    + [<span id="obs2">Observation Space</span>](#-span-id--obs2--observation-space--span-)
    + [<span id="action2">Action Space</span>](#-span-id--action2--action-space--span-)
    + [<span id="r2">Rewards</span>](#-span-id--r2--rewards--span-)
  * [HandCatchOver2Underarm Environments](#handcatchover2underarm-environments)
    + [<span id="obs3">Observation Space</span>](#-span-id--obs3--observation-space--span-)
    + [<span id="action3">Action Space</span>](#-span-id--action3--action-space--span-)
    + [<span id="r3">Rewards</span>](#-span-id--r3--rewards--span-)
  * [TwoObjectCatch Environments](#twoobjectcatch-environments)
    + [<span id="obs4">Observation Space</span>](#-span-id--obs4--observation-space--span-)
    + [<span id="action4">Action Space</span>](#-span-id--action4--action-space--span-)
    + [<span id="r4">Rewards</span>](#-span-id--r4--rewards--span-)
  * [HandCatchAbreast Environments](#handcatchabreast-environments)
    + [<span id="obs5">Observation Space</span>](#-span-id--obs5--observation-space--span-)
    + [<span id="action5">Action Space</span>](#-span-id--action5--action-space--span-)
    + [<span id="r5">Rewards</span>](#-span-id--r5--rewards--span-)
***


### About this repository

This repository is an extention of [DexterousHands](https://github.com/PKU-MARL/DexterousHands) which is from PKU MARL research team, Safe MAIG contains complex dexterous hand RL environments for the NVIDIA Isaac Gym high performance environments described in the NeurIPS 2021 Datasets and Benchmarks [paper.](https://openreview.net/forum?id=fgFBtYgJQX_)

:star2: This repository is under actively development. We appreciate any constructive comments and suggestions. If you have any questions, please feel free to email <gshangd[AT]foxmail.com>.

 <div align=center>
 <img src="https://github.com/chauncygu/Safe-Multi-Agent-Isaac-Gym/blob/main/docs/safe-multi-agent-isaac-gym.png" width="850"/> 
 </div>
<div align=center>
<center style="color:#000000;text-decoration:underline">Figure.1 Safe multi-agent Isaac Gym environments. Body parts of different colours of robots are controlled by different agents in each pair of hands. Agents jointly learn to manipulate the robot, while avoiding violating the safety constraints. </center>
 </div>



## Installation

Details regarding installation of IsaacGym can be found [here](https://developer.nvidia.com/isaac-gym). **We currently support the `Preview Release 3` version of IsaacGym.**

### Pre-requisites

The code has been tested on Ubuntu 18.04 with Python 3.7. The minimum recommended NVIDIA driver
version for Linux is `470` (dictated by support of IsaacGym).

It uses [Anaconda](https://www.anaconda.com/) to create virtual environments.
To install Anaconda, follow instructions [here](https://docs.anaconda.com/anaconda/install/linux/).

Ensure that Isaac Gym works on your system by running one of the examples from the `python/examples` 
directory, like `joint_monkey.py`. Follow troubleshooting steps described in the Isaac Gym Preview 2 
install instructions if you have any trouble running the samples.

### install this repo
Once Isaac Gym is installed and samples work within your current python environment, install this repo:

```bash
pip install -e .
```

## Running the benchmarks

To train your first policy, run this line:

```bash
python train.py --task=ShadowHandOver --algo=macpo
```
Or run the lines:

```bash
chmod +x ./run_experiments.sh
./run_experiments.sh
```

If you want to turn off the video, set headless as True



## Select an algorithm

To select an algorithm, pass `--algo=ppo/mappo/happo/hatrpo` 
as an argument:

```bash
python train.py --task=ShadowHandOver --algo=macpo
``` 

At present, we only support these four algorithms.
<!-- ### Loading trained models // Checkpoints

Checkpoints are saved in the folder `models/` 

To load a trained checkpoint and only perform inference (no training), pass `--test` 
as an argument:

```bash
python train.py --task=ShadowHandOver --checkpoint=models/shadow_hand_over/ShadowHandOver.pth --test
``` -->

<!--## <span id="task">Tasks</span>-->

## Select tasks

Source code for tasks can be found in `dexteroushandenvs/tasks`. 

Until now we only suppose the following environments:

| Environments | ShadowHandOver | ShadowHandCatchUnderarm | ShadowHandTwoCatchUnderarm | ShadowHandCatchAbreast | ShadowHandOver2Underarm |
|  :----:  | :----:  | :----:  | :----:  | :----:  | :----:  |
| Description | These environments involve two fixed-position hands. The hand which starts with the object must find a way to hand it over to the second hand. | These environments again have two hands, however now they have some additional degrees of freedom that allows them to translate/rotate their centre of masses within some constrained region. | These environments involve coordination between the two hands so as to throw the two objects between hands (i.e. swapping them). | This environment is similar to ShadowHandCatchUnderarm, the difference is that the two hands are changed from relative to side-by-side posture. | This environment is is made up of half ShadowHandCatchUnderarm and half ShadowHandCatchOverarm, the object needs to be thrown from the vertical hand to the palm-up hand |
| Actions Type | Continuous | Continuous | Continuous | Continuous | Continuous |
| Total Action Num | 40    | 52    | 52    | 52    | 52    |
| Action Values     | [-1, 1]    | [-1, 1]    | [-1, 1]    | [-1, 1]    | [-1, 1]    |
| Action Index and Description     | [detail](#action1)    | [detail](#action2)   | [detail](#action3)    | [detail](#action4)    | [detail](#action5)    |
| Observation Shape     | (num_envs, 2, 211)    | (num_envs, 2, 217)    | (num_envs, 2, 217)    | (num_envs, 2, 217)    | (num_envs, 2, 217)    |
| Observation Values     | [-5, 5]    | [-5, 5]    | [-5, 5]    | [-5, 5]    | [-5, 5]    |
| Observation Index and Description     | [detail](#obs1)    | [detail](#obs2)   | [detail](#obs3)    | [detail](#obs4)    | [detail](#obs4)    |
| State Shape     | (num_envs, 2, 398)    | (num_envs, 2, 422)    | (num_envs, 2, 422)    | (num_envs, 2, 422)    | (num_envs, 2, 422)    | 
| State Values     | [-5, 5]    | [-5, 5]    | [-5, 5]    | [-5, 5]    | [-5, 5]    |
| Rewards     | Rewards is the pose distance between object and goal. You can check out the details [here](#r1)| Rewards is the pose distance between object and goal. You can check out the details [here](#r2)    | Rewards is the pose distance between object and goal. You can check out the details [here](#r3)    | Rewards is the pose distance between two object and  two goal, this means that both objects have to be thrown in order to be swapped over. You can check out the details [here](#r4)    | Rewards is the pose distance between object and goal. You can check out the details [here](#r2)    |
| Demo     | <img src="assets/image_folder/0v1.gif" align="middle" width="150" border="1"/>    | <img src="assets/image_folder/hand_catch_underarm.gif" align="middle" width="140" border="1"/>    | <img src="assets/image_folder/two_catch.gif" align="middle" width="130" border="1"/>    | <img src="assets/image_folder/1v1.gif" align="middle" width="130" border="1"/>    | <img src="assets/image_folder/2.gif" align="middle" width="130" border="1"/>    |



### HandOver Environments
<img src="assets/image_folder/0v1.gif" align="middle" width="450" border="1"/>

These environments involve two fixed-position hands. The hand which starts with the object must find a way to hand it over to the second hand. To use the HandOver environment, pass `--task=ShadowHandOver`

#### <span id="obs1">Observation Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 23 | shadow hand dof position |
| 24 - 47     | shadow hand dof velocity    |
| 48 - 71     | shadow hand dof force    |
| 72 - 136    | shadow hand fingertip pose, linear velocity, angle velocity (5 x 13) |
| 137 - 166     | shadow hand fingertip force, torque (5 x 6)    |
| 167 - 186     | actions    |
| 187 - 193     | object pose   |
| 194 - 196     | object linear velocity    |
| 197 - 199     | object angle velocity    |
| 200 - 206     | goal pose    |
| 207 - 210     | goal rot - object rot   |

#### <span id="action1">Action Space</span>
The shadow hand has 24 joints, 20 actual drive joints and 4 underdrive joints. So our Action is the joint Angle value of the 20 dimensional actuated joint.
| Index | Description |
|  ----  | ----  |
| 0 - 19 | shadow hand actuated joint |

#### <span id="r1">Rewards</span>
Rewards is the pose distance between object and goal, and the specific formula is as follows:
```python
goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)

quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

dist_rew = goal_dist

reward = torch.exp(-0.2*(dist_rew * dist_reward_scale + rot_dist))
```

### HandCatchUnderarm Environments
<img src="assets/image_folder/hand_catch_underarm.gif" align="middle" width="450" border="1"/>

These environments again have two hands, however now they have some additional degrees of freedom that allows them to translate/rotate their centre of masses within some constrained region. To use the HandCatchUnderarm environment, pass `--task=ShadowHandCatchUnderarm`

#### <span id="obs2">Observation Space</span>


| Index | Description |
|  :----:  | :----:  |
| 0 - 23 | shadow hand dof position |
| 24 - 47     | shadow hand dof velocity    |
| 48 - 71     | shadow hand dof force    |
| 72 - 136    | shadow hand fingertip pose, linear velocity, angle velocity (5 x 13) |
| 137 - 166     | shadow hand fingertip force, torque (5 x 6)    |
| 167 - 192     | actions    |
| 193 - 195     | shadow hand transition    |
| 196 - 198     | shadow hand orientation    |
| 199 - 205     | object pose   |
| 206 - 208     | object linear velocity    |
| 209 - 211     | object angle velocity    |
| 212 - 218     | goal pose    |
| 219 - 222     | goal rot - object rot   |

#### <span id="action2">Action Space</span>

Similar to the HandOver environments, except now the bases are not fixed and have translational and rotational degrees of freedom that allow them to move within some range.
| Index | Description |
|  :----:  | :----:  |
| 0 - 19 | shadow hand actuated joint |
| 20 - 22 | shadow hand actor translation |
| 23 - 25 | shadow hand actor rotation |

#### <span id="r2">Rewards</span>

Rewards is the pose distance between object and goal, and the specific formula is as follows:
```python
goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)

quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

dist_rew = goal_dist

reward = torch.exp(-0.2*(dist_rew * dist_reward_scale + rot_dist))
```

### HandCatchOver2Underarm Environments
<img src="assets/image_folder/2.gif" align="middle" width="450" border="1"/>

This environment is is made up of half ShadowHandCatchUnderarm and half ShadowHandCatchOverarm, the object needs to be thrown from the vertical hand to the palm-up hand. To use the HandCatchUnderarm environment, pass `--task=ShadowHandCatchOver2Underarm`

#### <span id="obs3">Observation Space</span>


| Index | Description |
|  :----:  | :----:  |
| 0 - 23 | shadow hand dof position |
| 24 - 47     | shadow hand dof velocity    |
| 48 - 71     | shadow hand dof force    |
| 72 - 136    | shadow hand fingertip pose, linear velocity, angle velocity (5 x 13) |
| 137 - 166     | shadow hand fingertip force, torque (5 x 6)    |
| 167 - 192     | actions    |
| 193 - 195     | shadow hand transition    |
| 196 - 198     | shadow hand orientation    |
| 199 - 205     | object pose   |
| 206 - 208     | object linear velocity    |
| 209 - 211     | object angle velocity    |
| 212 - 218     | goal pose    |
| 219 - 222     | goal rot - object rot   |

#### <span id="action3">Action Space</span>

Similar to the HandOver environments, except now the bases are not fixed and have translational and rotational degrees of freedom that allow them to move within some range.
| Index | Description |
|  :----:  | :----:  |
| 0 - 19 | shadow hand actuated joint |
| 20 - 22 | shadow hand actor translation |
| 23 - 25 | shadow hand actor rotation |

#### <span id="r3">Rewards</span>

Rewards is the pose distance between object and goal, and the specific formula is as follows:
```python
goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
# Orientation alignment for the cube in hand and goal cube
quat_diff = quat_mul(object_rot, quat_conjugate(target_rot)
reward = (0.3 - goal_dist - quat_diff)
```

### TwoObjectCatch Environments
<img src="assets/image_folder/two_catch.gif" align="middle" width="450" border="1"/>

These environments involve coordination between the two hands so as to throw the two objects between hands (i.e. swapping them). This is necessary since each object's goal can only be reached by the other hand. To use the HandCatchUnderarm environment, pass `--task=ShadowHandTwoCatchUnderarm`

#### <span id="obs4">Observation Space</span>

| Index | Description |
|  :----:  | :----:  |
| 0 - 23 | shadow hand dof position |
| 24 - 47     | shadow hand dof velocity    |
| 48 - 71     | shadow hand dof force    |
| 72 - 136    | shadow hand fingertip pose, linear velocity, angle velocity (5 x 13) |
| 137 - 166     | shadow hand fingertip force, torque (5 x 6)    |
| 167 - 192     | actions    |
| 193 - 195     | shadow hand transition    |
| 196 - 198     | shadow hand orientation    |
| 199 - 205     | object1 pose   |
| 206 - 208     | object1 linear velocity    |
| 210 - 212     | object1 angle velocity    |
| 213 - 219     | goal1 pose    |
| 220 - 223     | goal1 rot - object1 rot   |
| 224 - 230     | object2 pose   |
| 231 - 233     | object2 linear velocity    |
| 234 - 236     | object2 angle velocity    |
| 237 - 243     | goal2 pose    |
| 244 - 247     | goal2 rot - object2 rot   |

#### <span id="action4">Action Space</span>

Similar to the HandOver environments, except now the bases are not fixed and have translational and rotational degrees of freedom that allow them to move within some range.
| Index | Description |
|  :----:  | :----:  |
| 0 - 19 | shadow hand actuated joint |
| 20 - 22 | shadow hand actor translation |
| 23 - 25 | shadow hand actor rotation |

#### <span id="r4">Rewards</span>

Rewards is the pose distance between two object and  two goal, this means that both objects have to be thrown in order to be swapped over. The specific formula is as follows:
```python
goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
goal_another_dist = torch.norm(target_another_pos - object_another_pos, p=2, dim=-1)

# Orientation alignment for the cube in hand and goal cube
quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

quat_another_diff = quat_mul(object_another_rot, quat_conjugate(target_another_rot))
rot_another_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_another_diff[:, 0:3], p=2, dim=-1), max=1.0))

dist_rew = goal_dist

reward = torch.exp(-0.2*(dist_rew * dist_reward_scale + rot_dist)) + torch.exp(-0.2*(goal_another_dist * dist_reward_scale + rot_another_dist))
```

### HandCatchAbreast Environments
<img src="assets/image_folder/1v1.gif" align="middle" width="450" border="1"/>

This environment is similar to ShadowHandCatchUnderarm, the difference is that the two hands are changed from relative to side-by-side posture.. To use the HandCatchAbreast environment, pass `--task=ShadowHandCatchAbreast`

#### <span id="obs5">Observation Space</span>


| Index | Description |
|  :----:  | :----:  |
| 0 - 23 | shadow hand dof position |
| 24 - 47     | shadow hand dof velocity    |
| 48 - 71     | shadow hand dof force    |
| 72 - 136    | shadow hand fingertip pose, linear velocity, angle velocity (5 x 13) |
| 137 - 166     | shadow hand fingertip force, torque (5 x 6)    |
| 167 - 192     | actions    |
| 193 - 195     | shadow hand transition    |
| 196 - 198     | shadow hand orientation    |
| 199 - 205     | object pose   |
| 206 - 208     | object linear velocity    |
| 209 - 211     | object angle velocity    |
| 212 - 218     | goal pose    |
| 219 - 222     | goal rot - object rot   |

#### <span id="action5">Action Space</span>

Similar to the HandOver environments, except now the bases are not fixed and have translational and rotational degrees of freedom that allow them to move within some range.
| Index | Description |
|  :----:  | :----:  |
| 0 - 19 | shadow hand actuated joint |
| 20 - 22 | shadow hand actor translation |
| 23 - 25 | shadow hand actor rotation |

#### <span id="r5">Rewards</span>

Rewards is the pose distance between object and goal, and the specific formula is as follows:
```python
goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)

quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

dist_rew = goal_dist

reward = torch.exp(-0.2*(dist_rew * dist_reward_scale + rot_dist))
```
