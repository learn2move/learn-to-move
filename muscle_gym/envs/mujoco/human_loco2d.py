import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

import os
import yaml
from gym import spaces
from muscle_gym.muscle import MusculoTendonJoint as MTJ
# ----------------------------------------------------------------------------------
class HumanLoco2dEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    qpos_keys = ['rootx', 'rootz', 'roottheta',
                'r_hip', 'r_knee', 'r_ankle',
                'l_hip', 'l_knee', 'l_ankle']
    
    R_LEG = 0
    L_LEG = 1

    #restructuring for readbility
    #trunk:
            # pos
    
    #right_leg:
            # joints
                #hip, knee, ankle

            # muscle_activations
                #HFL...
                    #v_ce, l_ce, F_mtu

            # contact
                #is_contact #bool
                # GRF #ground reaction force 
                # GRM #ground reaction moment
    #left_leg:
            # joints
                #hip, knee, ankle

            # muscle_activations
                #HFL
                    #v_ce, l_ce, F_mtu

            # contact
                #is_contact
                # GRF #ground reaction force 
                # GRM #ground reaction moment


    obs_keys = [
                'rootx', 'rootz', 'roottheta',
                'r_hip', 'r_knee', 'r_ankle',
                'l_hip', 'l_knee', 'l_ankle',
                
                'd_rootx', 'd_rootz', 'd_roottheta',
                'd_r_hip', 'd_r_knee', 'd_r_ankle',
                'd_l_hip', 'd_l_knee', 'd_l_ankle',
                
                'r_trunk_x', 'r_trunk_y', 'r_trunk_z',
                'r_trunk_dx', 'r_trunk_dy', 'r_trunk_dz',
                'r_contact', 
                'r_GRF_x', 'r_GRF_y', 'r_GRF_z',
                'r_GRM_x', 'r_GRM_y', 'r_GRM_z',

                'r_HFL_v_ce', 'r_HFL_l_ce', 'r_HFL_F_mtu',
                'r_GLU_v_ce', 'r_GLU_l_ce', 'r_GLU_F_mtu',
                'r_HAM_v_ce', 'r_HAM_l_ce', 'r_HAM_F_mtu',
                'r_RF_v_ce', 'r_RF_l_ce', 'r_RF_F_mtu',
                'r_VAS_v_ce', 'r_VAS_l_ce', 'r_VAS_F_mtu',
                'r_BFSH_v_ce', 'r_BFSH_l_ce', 'r_BFSH_F_mtu',
                'r_GAS_v_ce', 'r_GAS_l_ce', 'r_GAS_F_mtu',
                'r_SOL_v_ce', 'r_SOL_l_ce', 'r_SOL_F_mtu',
                'r_TA_v_ce', 'r_TA_l_ce', 'r_TA_F_mtu',
                
                'l_trunk_x', 'l_trunk_y', 'l_trunk_z',
                'l_trunk_dx', 'l_trunk_dy', 'l_trunk_dz',
                'l_contact', 
                'l_GRF_x', 'l_GRF_y', 'l_GRF_z',
                'l_GRM_x', 'l_GRM_y', 'l_GRM_z',

                'l_HFL_v_ce', 'l_HFL_l_ce', 'l_HFL_F_mtu',
                'l_GLU_v_ce', 'l_GLU_l_ce', 'l_GLU_F_mtu',
                'l_HAM_v_ce', 'l_HAM_l_ce', 'l_HAM_F_mtu',
                'l_RF_v_ce', 'l_RF_l_ce', 'l_RF_F_mtu',
                'l_VAS_v_ce', 'l_VAS_l_ce', 'l_VAS_F_mtu',
                'l_BFSH_v_ce', 'l_BFSH_l_ce', 'l_BFSH_F_mtu',
                'l_GAS_v_ce', 'l_GAS_l_ce', 'l_GAS_F_mtu',
                'l_SOL_v_ce', 'l_SOL_l_ce', 'l_SOL_F_mtu',
                'l_TA_v_ce', 'l_TA_l_ce', 'l_TA_F_mtu',
                ]

    # !!! todo:
    # don't use dictionary if unnecessary
    # use legs = ['r_leg', 'l_leg'], sides ['r', 'l'] (only one if possible)
    muscle_types = ['HFL', 'GLU', 'HAM', 'RF', 'VAS', 'BFSH', 'GAS', 'SOL', 'TA']
    qpos_map = dict(zip(qpos_keys, range(len(qpos_keys))))
    obs_map = dict(zip(obs_keys, range(len(obs_keys))))

    def __init__(self):
        self._import_model(filename='humanloco2d.xml', frameskip=1)
        self.total_muscles, self.muscle_joint = self._init_muscles(timestep=self.dt, muscle_inter_timestep=0.002)
        assert(self.total_muscles == 18)

        self.gravity = 9.81 # !!! todo: read from self.model (for self.sim)
        self.mass_total = sum(self.model.body_mass)
        self.length_leg = abs(self.model.body_pos[self.model.body_name2id('r_shank')][2]) \
                    + abs(self.model.body_pos[self.model.body_name2id('r_foot')][2])
                    # !!! todo: read segment leangths
        self.FORCE0 = self.mass_total*self.gravity
        self.LENGTH0 = self.length_leg

        self.t_end = 20
        self.v_tgt = 1.4

        self.reset_model() #sets action space and resets muscles
        return
    # -----------------------------------------------------------------------------------------------------------------
    def _import_model(self, filename, frameskip=4):
        dirname, _ = os.path.split(os.path.abspath(__file__))
        fullpath = os.path.join(dirname, 'assets', filename)
        mujoco_env.MujocoEnv.__init__(self, fullpath, frameskip)
        utils.EzPickle.__init__(self)
        return
    # -----------------------------------------------------------------------------------------------------------------
    def step(self, a):
        if not hasattr(self, 'muscles'):
            return np.zeros(1), 0, False, {}
        stim = a
        self.t += self.dt
        trq = self._muscle_step(stim)
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(trq, self.frame_skip)

        obs = self._get_obs()
        self.count_footstep(obs)
        done = self.fail_detection(obs)
        reward = self.get_reward(obs, stim, done)

        return obs, reward, done, {}

    # -----------------------------------------------------------------------------------------------------------------
    def _get_obs(self):
        obs = {} #switch to a dictionary
        # ------------------------------------------------
        # joint values
        qpos = self.sim.data.qpos

        #obs['rootx'] = qpos[0]
        #obs['rootz'] = qpos[1] # no need of global position
        obs['roottheta'] = qpos[2]
        
        # obs['r_hip'] = qpos[3]
        # obs['r_knee'] = qpos[4]
        # obs['r_ankle'] = qpos[5]

        # obs['l_hip'] = qpos[6]
        # obs['l_knee'] = qpos[7]
        # obs['l_ankle'] = qpos[8]
        # ------------------------------------------------
        # derivatives of joint values
        qvel = self.sim.data.qvel

        obs['d_rootx'] = qvel[0]/self.LENGTH0
        obs['d_rootz'] = qvel[1]/self.LENGTH0
        obs['d_roottheta'] = qvel[2]
        
        # obs['d_r_hip'] = qvel[3]
        # obs['d_r_knee'] = qvel[4]
        # obs['d_r_ankle'] = qvel[5]

        # obs['d_l_hip'] = qvel[6]
        # obs['d_l_knee'] = qvel[7]
        # obs['d_l_ankle'] = qvel[8]
        # ------------------------------------------------
        # muscle state
        for s_l in ['r', 'l']: # leg
            s_leg = '{}_leg'.format(s_l)
            for s_m in self.muscle_types:
                for s_s in ['v_ce', 'l_ce', 'F_mtu']:
                    obs['{}_{}_{}'.format(s_l, s_m, s_s)] \
                        = self.muscles[s_leg][s_m].getSensoryData(s_s)

        # ------------------------------------------------
        # Notation information
        # self.sim.data.cfrc_ext #8 body parts x 6(rotation, translation)
        # world, torso, r_thigh, r_shank, r_foot, l_thigh, l_shank, l_foot
        # we focus on foots
        r_cfrc_ext = self.sim.data.cfrc_ext[4]
        l_cfrc_ext = self.sim.data.cfrc_ext[7]

        # ------------------------------------------------
        # ground reaction moment
        obs['r_GRM_x'] = r_cfrc_ext[0]/(self.FORCE0*self.LENGTH0)
        obs['r_GRM_y'] = r_cfrc_ext[1]/(self.FORCE0*self.LENGTH0)
        obs['r_GRM_z'] = r_cfrc_ext[2]/(self.FORCE0*self.LENGTH0)
        
        obs['l_GRM_x'] = l_cfrc_ext[0]/(self.FORCE0*self.LENGTH0)
        obs['l_GRM_y'] = l_cfrc_ext[1]/(self.FORCE0*self.LENGTH0)
        obs['l_GRM_z'] = l_cfrc_ext[2]/(self.FORCE0*self.LENGTH0)
        
        # ------------------------------------------------
        # ground reaction force
        obs['r_GRF_x'] = r_cfrc_ext[3]/(self.FORCE0)
        obs['r_GRF_y'] = r_cfrc_ext[4]/(self.FORCE0)
        obs['r_GRF_z'] = r_cfrc_ext[5]/(self.FORCE0)

        obs['l_GRF_x'] = l_cfrc_ext[3]/(self.FORCE0)
        obs['l_GRF_y'] = l_cfrc_ext[4]/(self.FORCE0)
        obs['l_GRF_z'] = l_cfrc_ext[5]/(self.FORCE0)

        # ------------------------------------------------
        # is in contact with ground
        obs['r_contact'] = True if obs['r_GRF_z'] > 0 else False 
        obs['l_contact'] = True if obs['l_GRF_z'] > 0 else False

        # ------------------------------------------------

        # right leg to trunk distance
        trunk_pos = self.get_body_com("torso") #[x,y,z]
        r_foot_pos = self.get_body_com("r_foot") #[x,y,z]
        r_trunk_pos = trunk_pos - r_foot_pos #vector from trunk to right leg
        obs['r_trunk_x'] = r_trunk_pos[0]/(self.LENGTH0)
        obs['r_trunk_y'] = r_trunk_pos[1]/(self.LENGTH0)
        obs['r_trunk_z'] = r_trunk_pos[2]/(self.LENGTH0)

        # ------------------------------------------------
        # left leg to trunk distance
        l_foot_pos = self.get_body_com("l_foot") #[x,y,z]
        l_trunk_pos = trunk_pos - l_foot_pos #vector from trunk to right leg
        obs['l_trunk_x'] = l_trunk_pos[0]/(self.LENGTH0)
        obs['l_trunk_y'] = l_trunk_pos[1]/(self.LENGTH0)
        obs['l_trunk_z'] = l_trunk_pos[2]/(self.LENGTH0)
        
        # ------------------------------------------------
        # right leg to trunk velocity
        obs['r_trunk_dx'] = (obs['r_trunk_x'] - self.prev_obs['r_trunk_x'])/self.dt
        obs['r_trunk_dy'] = (obs['r_trunk_y'] - self.prev_obs['r_trunk_y'])/self.dt
        obs['r_trunk_dz'] = (obs['r_trunk_z'] - self.prev_obs['r_trunk_z'])/self.dt

        # ------------------------------------------------
        # left leg to trunk velocity
        obs['l_trunk_dx'] = (obs['l_trunk_x'] - self.prev_obs['l_trunk_x'])/self.dt
        obs['l_trunk_dy'] = (obs['l_trunk_y'] - self.prev_obs['l_trunk_y'])/self.dt
        obs['l_trunk_dz'] = (obs['l_trunk_z'] - self.prev_obs['l_trunk_z'])/self.dt
        # ------------------------------------------------

        self.prev_obs = obs
        return obs
    # -----------------------------------------------------------------------------------------------------------------

    def count_footstep(self, obs):
        contact = np.empty(2) # foot in conact
        contact[self.R_LEG] = obs['r_contact']
        contact[self.L_LEG] = obs['l_contact']

        for i_leg in [self.R_LEG, self.L_LEG]:
            if not self.in_contact[i_leg] and contact[i_leg] and self.last_HS is not i_leg:
                self.n_step += 1
                self.last_HS = i_leg
                self.t_new_step = self.t

        self.in_contact[self.R_LEG] = contact[self.R_LEG]
        self.in_contact[self.L_LEG] = contact[self.L_LEG]
    # -----------------------------------------------------------------------------------------------------------------

    def get_reward(self, obs, stim, done):
        self.ACT2_total += sum(np.square(stim))*self.dt

        # alive bonus
        # should be large enough to search for 'success' solutions (alive to the end) first
        reward = .1

        # footstep reward (when made a new step)
        reward_footstep = 0
        if self.t_new_step != self.footstep['t']:
            # update footstep data
            self.footstep['del_t'] = self.t_new_step - self.footstep['t']
            self.footstep['t'] = self.t_new_step
            self.footstep['del_x'] = self.sim.data.qpos[0] - self.footstep['x']
            self.footstep['x'] = self.sim.data.qpos[0]
            self.footstep['FOT'] = (self.ACT2_total - self.footstep['ACT2_total'])/self.footstep['del_x']
            self.footstep['ACT2_total'] = self.ACT2_total
            self.footstep['v'] = self.footstep['del_x']/self.footstep['del_t']

            # footstep reward: so that solution does not avoid making footsteps
            # scaled by del_t, so that solution does get higher rewards by making unnecessary (small) steps
            reward_footstep += self.reward_w['footstep']*self.footstep['del_t']

            # panalize FOT (fatigue of transport)
            # FOT = (muscle fatigure)/(traveled distance)
            # comparable to the COT (cost of transport)
            reward_footstep -= self.reward_w['FOT']*self.footstep['FOT']

            # deviation from target velocity
            # the average velocity a step (instead of instantaneous velocity) is used
            # as velocity fluctuates within a step in normal human walking
            reward_footstep -= self.reward_w['v_tgt']*np.abs(self.footstep['v']-self.v_tgt)*self.footstep['del_t']

        # success bonus
        if done and self.failure_mode is 'success':
            reward += 1000 + 10*reward_footstep
        else:
            reward += reward_footstep

        return reward
    # -----------------------------------------------------------------------------------------------------------------

    def fail_detection(self, obs):
        contact = np.empty(2) # foot in conact
        contact[self.R_LEG] = obs['r_contact']
        contact[self.L_LEG] = obs['l_contact']
        rel_height = np.empty(2) # trunk height relative to foot
        rel_height[self.R_LEG] = obs['r_trunk_z']
        rel_height[self.L_LEG] = obs['l_trunk_z']
        theta = obs['roottheta']

        done = 0
        for i_leg in [self.R_LEG, self.L_LEG]:
            if contact[i_leg] and rel_height[i_leg] < 0.3:
                done = 1
                self.failure_mode = 'stance leg collapse'
        if theta > 1.0:
            done = 1
            self.failure_mode = 'forward fall'
        if theta < -1.0:
            done = 1
            self.failure_mode = 'backward fall'
        if self.t_new_step + 3.0 < self.t:
            done = 1
            self.failure_mode = 'no more step'

        if self.t > self.t_end:
            done = 1
            self.failure_mode = 'success'
        return done

    # -----------------------------------------------------------------------------------------------------------------
    def reset_model(self):
        # qpos = self.init_qpos
        # qvel = self.init_qvel
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)

        #-----set action space cardinality---------
        low = np.zeros(self.total_muscles)
        high = np.ones(self.total_muscles)
        self.action_space = spaces.Box(low, high) #this updates the action space of the model

        #------reset muscles-----------------
        for s_l in ["r", "l"]:
            s_leg = "{}_leg".format(s_l)
            for muscle_type in self.muscle_types:
                muscle_name = "{}_{}".format(s_l, muscle_type)
                if len(self.muscle_joint[muscle_name]) == 1:
                    phi1 = self.sim.data.qpos[self.qpos_map[self.muscle_joint[muscle_name][0]]]
                    self.muscles[s_leg][muscle_type].reset(phi1)
                elif len(self.muscle_joint[muscle_name]) == 2:
                    phi1 = self.sim.data.qpos[self.qpos_map[self.muscle_joint[muscle_name][0]]]
                    phi2 = self.sim.data.qpos[self.qpos_map[self.muscle_joint[muscle_name][1]]]
                    self.muscles[s_leg][muscle_type].reset(phi1, phi2)
        
        #-----reset prev_obs variable--------
        self.prev_obs = {}
        self.prev_obs['r_trunk_x'] = 0
        self.prev_obs['r_trunk_y'] = 0
        self.prev_obs['r_trunk_z'] = 0

        self.prev_obs['l_trunk_x'] = 0
        self.prev_obs['l_trunk_y'] = 0
        self.prev_obs['l_trunk_z'] = 0

        self.t = 0
        self.failure_mode = None
        self.n_step = 0
        self.t_new_step = self.t
        self.in_contact = np.ones(2)
        self.in_contact[self.R_LEG] = 1
        self.in_contact[self.L_LEG] = 0
        self.last_HS = self.R_LEG # simulation starts with R_LEG as last heel strike
        self.ACT2_total = 0

        self.footstep = {}
        self.footstep['t'] = self.t_new_step
        self.footstep['x'] = self.sim.data.qpos[0]
        self.footstep['ACT2_total'] = 0

        self.reward_w = {}
        self.reward_w['footstep'] = 20
        self.reward_w['FOT'] = 1
        self.reward_w['v_tgt'] = 10

        return self._get_obs()

    # -----------------------------------------------------------------------------------------------------------------
    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = .8
        self.viewer.cam.elevation = -20
        self.viewer.cam.type = 1

    # -----------------------------------------------------------------------------------------------------------------
    def _muscle_step(self, stim):
        joint_trq = {
            'r_hip': 0,
            'r_knee': 0,
            'r_ankle':0,
            'l_hip': 0,
            'l_knee': 0,
            'l_ankle': 0
            }
        i_muscle = 0
        for s_l in ["r", "l"]:
            s_leg = "{}_leg".format(s_l)
            for muscle_type in self.muscle_types:
                muscle_name = "{}_{}".format(s_l, muscle_type)
                if len(self.muscle_joint[muscle_name]) == 1:
                    phi1 = self.sim.data.qpos[self.qpos_map[self.muscle_joint[muscle_name][0]]]
                    self.muscles[s_leg][muscle_type].update(stim[i_muscle], phi1)
                    joint_trq[self.muscle_joint[muscle_name][0]] += self.muscles[s_leg][muscle_type].getTorque()
                elif len(self.muscle_joint[muscle_name]) == 2:
                    phi1 = self.sim.data.qpos[self.qpos_map[self.muscle_joint[muscle_name][0]]]
                    phi2 = self.sim.data.qpos[self.qpos_map[self.muscle_joint[muscle_name][1]]]
                    self.muscles[s_leg][muscle_type].update(stim[i_muscle], phi1, phi2)
                    joint_trq[self.muscle_joint[muscle_name][0]] += self.muscles[s_leg][muscle_type].getTorque()
                    joint_trq[self.muscle_joint[muscle_name][1]] += self.muscles[s_leg][muscle_type].getTorque2()
                i_muscle += 1
        trq = [
            joint_trq['r_hip'],
            joint_trq['r_knee'],
            joint_trq['r_ankle'],
            joint_trq['l_hip'],
            joint_trq['l_knee'],
            joint_trq['l_ankle']
        ]
        trq = np.asarray(trq)
        return trq
        
    # -----------------------------------------------------------------------------------------------------------------
    def _import_muscle_model(self, filename='humanloco2d.yml'):
        dirname, _ = os.path.split(os.path.abspath(__file__))
        fullpath = os.path.join(dirname, 'muscle_assets', filename)
        with open(fullpath, 'r') as f:
            muscle_cfg = yaml.load(f)
        return muscle_cfg

    def set_init_pos_r_stance(self, r_angles, l_alpha = 120*np.pi/180):
        # r_angles: [r_hip, r_knee, r_ankle]
        # todo: read from xml
        l_torso_0 = .5*0.425 + 0.075
        l_thigh = 0.44
        l_shank = 0.44
        h_foot = 0.05 + 0.03 # distance between sole and ankle
        l_foot1 = 0.16 + 0.03 # x_distance between ankle and toe tip
        x_pose = 0
        theta = -r_angles[2] + r_angles[1] - r_angles[0]
        h_r_leg = h_foot + l_shank*np.cos(r_angles[2]) + l_thigh*np.cos(r_angles[2] - r_angles[1])
        z_pose = h_r_leg + l_torso_0*np.cos(r_angles[2] - r_angles[1] + r_angles[0])
        
        # left leg configuration
        l_ankle = 0
        h_l_leg2 = h_r_leg - l_foot1 # z_distance from l_hip to l_ankle
        # np.sin(alpha)*2*l_shank*np.cos(.5*l_knee) == h_l_leg2 # assuming l_thigh == l_shank
        l_knee = -2*np.arccos(np.minimum(1., h_l_leg2/(np.sin(l_alpha)*2*l_shank)))
        l_hip = l_alpha - .5*np.pi - theta + .5*l_knee
        l_angles = np.array([l_hip, l_knee, l_ankle])
        angles = np.append(r_angles, l_angles)
        self.init_qpos = np.append([x_pose, z_pose, theta], angles)

    # -----------------------------------------------------------------------------------------------------------------
    def _init_muscles(self, timestep=0.008, muscle_inter_timestep=0.002):
        self.muscles = {}
        self.muscle_dt = timestep
        self.n_update_muscle = round(timestep/muscle_inter_timestep)

        total_muscles = 0
        muscle_joint = {}

        muscle_cfg = self._import_muscle_model(filename='humanloco2d.yml')

        for s_l in ['r', 'l']:
            muscles = {}
            for muscle_type in self.muscle_types:
                cfg = muscle_cfg[muscle_type]
                
                muscle_name = "{}_{}".format(s_l, muscle_type)
                muscle_joint[muscle_name] = ['{}_{}'.format(s_l, joint) for joint in cfg['JOINT']]
                
                mtu_par = muscle_cfg[muscle_type]['MTU_PAR'] #note the angle here are in degrees
                mtu_par['A'] = 0.01
                mtu_par['flag_afferent'] = 1
                # phi1_0 and phi2_0 handled in muscle.reset()
                mtu_par['phi1_0'] = 0
                if len(muscle_joint[muscle_name]) == 2:
                    mtu_par['phi2_0'] = 0
                muscles[muscle_type] = MTJ(self.muscle_dt, mtu_par, n_update=self.n_update_muscle)
                
                total_muscles += 1

            self.muscles["{}_leg".format(s_l)] = muscles
        return total_muscles, muscle_joint

# ----------------------------------------------------------------------------]