import gym
import sys
import numpy as np
from muscle_gym.control import LocoCtrl

# ------------------------------------------------
# env_name = 'MuscleHopper-v2'
env_name = 'MuscleHumanLoco2d-v2'
# ------------------------------------------------

done = False
render = True
# ------------------------------------------------
env = gym.make(env_name)

# env.env.init_qpos = np.array([0., 1.17, 15*np.pi/180, 
#     -30*np.pi/180, -4*np.pi/180, 17*np.pi/180,
#     -15*np.pi/180, -50*np.pi/180, -5*np.pi/180])
env.env.init_qpos = np.array([0., 1.17, 15*np.pi/180, 
    -30*np.pi/180, -4*np.pi/180, 17*np.pi/180,
    10*np.pi/180, -20*np.pi/180, -5*np.pi/180])
env.env.init_qvel[0] = 1.5

env.reset()

# # uncomment to run openai gym hopper
# env = gym.make('Hopper-v2')
# env.reset()
# --------------------------------------------------------------------

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

print("Env:{} State Space:{} Action Space:{}".format(env_name, state_dim, action_dim))

# ------------------------------------------------


ctrl_dt = env.env.dt
ctrl = LocoCtrl(ctrl_dt, control_mode=1)

t_check = 0
spinal_control_phase = {}
spinal_control_phase['r_leg'] = {}
spinal_control_phase['l_leg'] = {}
for phase in ctrl.spinal_control_phase['r_leg']:
	spinal_control_phase['r_leg'][phase] = []
	spinal_control_phase['l_leg'][phase] = []
r_GRF_z = []
r_GRF_z1 = []
l_GRF_z = []
l_GRF_z1 = []
save_stim = np.empty((0,18))
while(True):
	# action = env.action_space.sample()
	# action = np.zeros(18)
	# print(action)

	obs = env.env._get_obs()

	r_GRF_z1.append(obs['r_GRF_z'])
	l_GRF_z1.append(obs['l_GRF_z'])
	
	stim = ctrl.update(obs)

	# stim[9+2] = 1 # L HAM
	# stim[9+5] = 1 # L BFSH
	_, reward, done, _ = env.step(stim)
	# next_state, reward, done, _ = env.step(np.array([1, 1, 1, 1, 1, 1]))
	# next_state, reward, done, _ = env.step(np.array([0, 0, 0, 0, 0, 0]))
	#next_state, reward, done, _ = env.step(np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3]))
	# next_state, reward, done, _ = env.step(np.array([0.3, 0.3, 0.3, 0.3, 1.0, 0]))

	if render:
		env.render()

	for phase in spinal_control_phase['r_leg']:
		for s_leg in ['r_leg', 'l_leg']:
			spinal_control_phase[s_leg][phase].append(ctrl.spinal_control_phase[s_leg][phase])
	r_GRF_z.append(ctrl.sensor_data['r_leg']['load_ipsi'])
	l_GRF_z.append(ctrl.sensor_data['l_leg']['load_ipsi'])
	save_stim = np.concatenate((save_stim,np.array([stim])), axis=0)
	if ctrl.t > t_check:
		# input()
		t_check += .2

	if ctrl.t > 5.5:
		break
	# if done:
	# 	break
# ------------------------------------------------
import matplotlib.pyplot as plt
plt.subplot(len(spinal_control_phase['r_leg']), 2, 1)
plt.plot(spinal_control_phase['r_leg']['ph_st'])
plt.subplot(len(spinal_control_phase['r_leg']), 2, 2)
plt.plot(spinal_control_phase['l_leg']['ph_st'])

plt.subplot(len(spinal_control_phase['r_leg']), 2, 3)
plt.plot(spinal_control_phase['r_leg']['ph_st_csw'])
plt.subplot(len(spinal_control_phase['r_leg']), 2, 4)
plt.plot(spinal_control_phase['l_leg']['ph_st_csw'])

plt.subplot(len(spinal_control_phase['r_leg']), 2, 5)
plt.plot(spinal_control_phase['r_leg']['ph_st_sw0'])
plt.subplot(len(spinal_control_phase['r_leg']), 2, 6)
plt.plot(spinal_control_phase['l_leg']['ph_st_sw0'])

plt.subplot(len(spinal_control_phase['r_leg']), 2, 7)
plt.plot(spinal_control_phase['r_leg']['ph_st_st'])
plt.subplot(len(spinal_control_phase['r_leg']), 2, 8)
plt.plot(spinal_control_phase['l_leg']['ph_st_st'])

plt.subplot(len(spinal_control_phase['r_leg']), 2, 9)
plt.plot(spinal_control_phase['r_leg']['ph_sw'])
plt.subplot(len(spinal_control_phase['r_leg']), 2, 10)
plt.plot(spinal_control_phase['l_leg']['ph_sw'])

plt.subplot(len(spinal_control_phase['r_leg']), 2, 11)
plt.plot(spinal_control_phase['r_leg']['ph_sw_flex_k'])
plt.subplot(len(spinal_control_phase['r_leg']), 2, 12)
plt.plot(spinal_control_phase['l_leg']['ph_sw_flex_k'])

plt.subplot(len(spinal_control_phase['r_leg']), 2, 13)
plt.plot(spinal_control_phase['r_leg']['ph_sw_hold_k'])
plt.subplot(len(spinal_control_phase['r_leg']), 2, 14)
plt.plot(spinal_control_phase['l_leg']['ph_sw_hold_k'])

plt.subplot(len(spinal_control_phase['r_leg']), 2, 15)
plt.plot(spinal_control_phase['r_leg']['ph_sw_stop_l'])
plt.subplot(len(spinal_control_phase['r_leg']), 2, 16)
plt.plot(spinal_control_phase['l_leg']['ph_sw_stop_l'])

plt.subplot(len(spinal_control_phase['r_leg']), 2, 17)
plt.plot(spinal_control_phase['r_leg']['ph_sw_hold_l'])
plt.subplot(len(spinal_control_phase['r_leg']), 2, 18)
plt.plot(spinal_control_phase['l_leg']['ph_sw_hold_l'])

plt.figure()
plt.subplot(9,2,1)
plt.plot(save_stim[:,0])
plt.title('HFL')
plt.ylim(0, 1)
plt.subplot(9,2,2)
plt.plot(save_stim[:,9])
plt.title('HFL')
plt.ylim(0, 1)
plt.subplot(9,2,3)
plt.plot(save_stim[:,1])
plt.title('GLU')
plt.ylim(0, 1)
plt.subplot(9,2,4)
plt.plot(save_stim[:,10])
plt.title('GLU')
plt.ylim(0, 1)
plt.subplot(9,2,5)
plt.plot(save_stim[:,2])
plt.title('HAM')
plt.ylim(0, 1)
plt.subplot(9,2,6)
plt.plot(save_stim[:,11])
plt.title('HAM')
plt.ylim(0, 1)
plt.subplot(9,2,7)
plt.plot(save_stim[:,3])
plt.title('RF')
plt.ylim(0, 1)
plt.subplot(9,2,8)
plt.plot(save_stim[:,12])
plt.title('RF')
plt.ylim(0, 1)
plt.subplot(9,2,9)
plt.plot(save_stim[:,4])
plt.title('VAS')
plt.ylim(0, 1)
plt.subplot(9,2,10)
plt.plot(save_stim[:,13])
plt.title('VAS')
plt.ylim(0, 1)
plt.subplot(9,2,11)
plt.plot(save_stim[:,5])
plt.title('BFSH')
plt.ylim(0, 1)
plt.subplot(9,2,12)
plt.plot(save_stim[:,14])
plt.title('BFSH')
plt.ylim(0, 1)
plt.subplot(9,2,13)
plt.plot(save_stim[:,6])
plt.title('GAS')
plt.ylim(0, 1)
plt.subplot(9,2,14)
plt.plot(save_stim[:,15])
plt.title('GAS')
plt.ylim(0, 1)
plt.subplot(9,2,15)
plt.plot(save_stim[:,7])
plt.title('SOL')
plt.ylim(0, 1)
plt.subplot(9,2,16)
plt.plot(save_stim[:,16])
plt.title('SOL')
plt.ylim(0, 1)
plt.subplot(9,2,17)
plt.plot(save_stim[:,8])
plt.title('TA')
plt.ylim(0, 1)
plt.subplot(9,2,18)
plt.plot(save_stim[:,17])
plt.title('TA')
plt.ylim(0, 1)


plt.figure()
plt.subplot(2,2,1)
plt.plot(r_GRF_z)
plt.subplot(2,2,2)
plt.plot(l_GRF_z)
plt.subplot(2,2,3)
plt.plot(r_GRF_z1)
plt.subplot(2,2,4)
plt.plot(l_GRF_z1)

plt.show()
