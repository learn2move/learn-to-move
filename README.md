# Learn-To-Move
Muscle models for Mujoco models in OpenAI Gym

## Requirements
- python 3.5.5 (python 3.6 not supported)  
- mjpro 150    
- mujoco-py (>=1.50.1.56)
    1. `git clone git@github.com:openai/mujoco-py.git`
    2. `cd mujoco-py && python setup.py install`   
- gym (>=0.10.5)
    1. `git clone git@github.com:openai/gym.git`
    2. `cd gym && python setup.py install` 


## Installation
1. `git clone git@github.com:rawalkhirodkar/human_locomotion.git`  
2. `cd human_locomotion && python setup.py install`  

## Environment Names
Name Format: Muscle'AgentName'-v'version_number'   
- MuscleHumanLoco2d-v2 


## Usage
This code snippet will import the muscle model.  
`import muscle_gym, gym`  
`env_name = 'MuscleHumanLoco2d-v2'`  
`gym.make(env_name`  

## Test   
Refer to test folder's readme.txt for examples.


## Directory Structure
muscle_gym
	- control: contains logic for locomotive control for the agent
	- envs
		- mujoco: all mujoco based envs
	- muscle: logic for muscle behavior of the agent
