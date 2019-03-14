import gym
import gym.spaces
import sys
import numpy as np
import muscle_gym
# --------------------------------------------------------------------

###uncomment to run any env

##muscle agents
env_name = 'MuscleHumanLoco2d-v2'

# --------------------------------------
##gym agents
# env_name = 'Hopper-v2'
# env_name = 'Walker2d-v2'
# env_name = 'HalfCheetah-v2'
# env_name = 'Ant-v2'
# env_name = 'Humanoid-v2'
# env_name = 'HumanoidStandup-v2'
# env_name = 'Swimmer-v2'


# --------------------------------------------------------------------
env = gym.make(env_name)
env.reset()

# --------------------------------------------------------------------
done = False
render = True
# --------------------------------------------------------------------

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

print("Env:{} State Space:{} Action Space:{}".format(env_name, state_dim, action_dim))

# --------------------------------------------------------------------

while(True):
	action = env.action_space.sample()
	next_state, reward, done, _ = env.step(action)
	print(done)
	print('---------------------')

	if render:
		env.render()

	if done:
		break
# --------------------------------------------------------------------