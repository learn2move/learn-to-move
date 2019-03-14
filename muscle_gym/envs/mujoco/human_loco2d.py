import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

import os
import yaml
from gym import spaces
from muscle_gym.muscle import MusculoTendonJoint as MTJ
from collections import OrderedDict
import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------
class HumanLoco2dEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    qpos_keys = ['rootx', 'rootz', 'roottheta',
                'r_hip', 'r_knee', 'r_ankle',
                'l_hip', 'l_knee', 'l_ankle']
    

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
                'r_is_contact', 
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
                'l_is_contact', 
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
        trq = self._muscle_step(a)
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(trq, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        return ob, reward, done, {}

    # -----------------------------------------------------------------------------------------------------------------
    def _get_obs(self):
        obs = {} #switch to a dictionary
        # ------------------------------------------------
        # joint values
        qpos = self.sim.data.qpos

        #obs['rootx'] = qpos[0]
        #obs['rootz'] = qpos[1] # no need of global position
        obs['roottheta'] = qpos[2]
        
        obs['r_hip'] = qpos[3]
        obs['r_knee'] = qpos[4]
        obs['r_ankle'] = qpos[5]

        obs['l_hip'] = qpos[6]
        obs['l_knee'] = qpos[7]
        obs['l_ankle'] = qpos[8]
        # ------------------------------------------------
        # derivatives of joint values
        qvel = self.sim.data.qvel

        obs['d_rootx'] = qvel[0]/self.LENGTH0
        obs['d_rootz'] = qvel[1]/self.LENGTH0
        obs['d_roottheta'] = qvel[2]
        
        obs['d_r_hip'] = qvel[3]
        obs['d_r_knee'] = qvel[4]
        obs['d_r_ankle'] = qvel[5]

        obs['d_l_hip'] = qvel[6]
        obs['d_l_knee'] = qvel[7]
        obs['d_l_ankle'] = qvel[8]
        # ------------------------------------------------
        # muscle activations
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
        obs['r_is_contact'] = True if obs['r_GRF_z'] > 0 else False 
        obs['l_is_contact'] = True if obs['l_GRF_z'] > 0 else False

        # ------------------------------------------------

        # right leg to trunk distance
        trunk_pos = self.get_body_com("torso") #[x,y,z]
        r_pos = self.get_body_com("r_foot") #[x,y,z]
        r_trunk_pos = r_pos - trunk_pos #vector from trunk to right leg
        obs['r_trunk_x'] = r_trunk_pos[0]/(self.LENGTH0)
        obs['r_trunk_y'] = r_trunk_pos[1]/(self.LENGTH0)
        obs['r_trunk_z'] = r_trunk_pos[2]/(self.LENGTH0)

        # ------------------------------------------------
        # left leg to trunk distance
        l_pos = self.get_body_com("l_foot") #[x,y,z]
        l_trunk_pos = l_pos - trunk_pos #vector from trunk to right leg
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

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        #qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        #qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
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

        return self._get_obs()

    # -----------------------------------------------------------------------------------------------------------------
    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = .8
        self.viewer.cam.elevation = -20

    # -----------------------------------------------------------------------------------------------------------------
    def _muscle_step(self, a):
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
                    self.muscles[s_leg][muscle_type].update(a[i_muscle], phi1)
                    joint_trq[self.muscle_joint[muscle_name][0]] += self.muscles[s_leg][muscle_type].getTorque()
                elif len(self.muscle_joint[muscle_name]) == 2:
                    phi1 = self.sim.data.qpos[self.qpos_map[self.muscle_joint[muscle_name][0]]]
                    phi2 = self.sim.data.qpos[self.qpos_map[self.muscle_joint[muscle_name][1]]]
                    self.muscles[s_leg][muscle_type].update(a[i_muscle], phi1, phi2)
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
        print('====================')
        print('HFL: {}'.format(self.muscles['r_leg']['HFL'].getTorque()))
        print('GLU: {}'.format(self.muscles['r_leg']['GLU'].getTorque()))
        print('HAM: {}'.format(self.muscles['r_leg']['HAM'].getTorque()))
        print('RF: {}'.format(self.muscles['r_leg']['RF'].getTorque()))
        print('hip: {}'.format(
            self.muscles['r_leg']['HFL'].getTorque()
            + self.muscles['r_leg']['GLU'].getTorque()
            + self.muscles['r_leg']['HAM'].getTorque()
            + self.muscles['r_leg']['RF'].getTorque()))
        print('HAM: {}'.format(self.muscles['r_leg']['HAM'].getTorque2()))
        print('RF: {}'.format(self.muscles['r_leg']['RF'].getTorque2()))
        print('VAS: {}'.format(self.muscles['r_leg']['VAS'].getTorque()))
        print('BFSH: {}'.format(self.muscles['r_leg']['BFSH'].getTorque()))
        print('GAS: {}'.format(self.muscles['r_leg']['GAS'].getTorque()))
        print('knee: {}'.format(
            self.muscles['r_leg']['HAM'].getTorque2()
            + self.muscles['r_leg']['RF'].getTorque2()
            + self.muscles['r_leg']['VAS'].getTorque()
            + self.muscles['r_leg']['BFSH'].getTorque()
            + self.muscles['r_leg']['GAS'].getTorque()))
        trq = np.asarray(trq)
        print(trq)
        return trq
        
    # -----------------------------------------------------------------------------------------------------------------
    def _import_muscle_model(self, filename='humanloco2d.yml'):
        dirname, _ = os.path.split(os.path.abspath(__file__))
        fullpath = os.path.join(dirname, 'muscle_assets', filename)
        with open(fullpath, 'r') as f:
            muscle_cfg = yaml.load(f)
        return muscle_cfg

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