# Author(s): Seungmoon Song <seungmoon.song@gmail.com>
"""
adapted from:
- Song and Geyer. "A neural circuitry that emphasizes
spinal feedback generates diverse behaviours of human locomotion." The
Journal of physiology, 2015.
"""

from __future__ import division # '/' always means non-truncating division
import numpy as np


class LocoCtrl(object):
    DEBUG = 0

    RIGHT = 0 #<- r_leg
    LEFT = 1 #<- l_leg

    # (todo) use these when handling angles
    # THETA0 = 0*np.pi/180 # trunk angle when standing straight
    # S_THETA = 1 # 1: leaning forward > 0; -1: leaning backward > 0
    # HIP0 = 0*np.pi/180 # hip angle when standing straight
    # S_HIP = 1 # 1: extension > 0; -1: flexion > 0
    # KNEE0 = 0*np.pi/180 # knee angle when standing straight
    # S_KNEE = 1 # 1: extension > 0; -1: flexion > 0
    # ANKLE0 = 0*np.pi/180 # ankle angle when standing straight
    # S_ANKLE = 1 # 1: plantar flexion > 0; -1: dorsiflexion > 0

    # muscle names
    m_keys = ['HAB', 'HAD', 'HFL', 'GLU', 'HAM', 'RF', 'VAS', 'BFSH', 'GAS', 'SOL', 'TA']
    # body sensor data
    s_b_keys = ['theta', 'dx' 'dy' 'dtheta']
    # leg sensor data
    s_l_keys = ['trunk_x', 'trunk_y', 'trunk_z', 'trunk_dx', 'trunk_dy', 'trunk_dz',
        'contact', 'load_ipsi', 'load_contra', 'S_CHFL', 'S_CGLU', 'S_CHAM', 'S_CRF',
        'vce_HAM', 'vce_RF', 'vce_VAS', 'vce_BFSH'
        'lce_HFL', 'lce_GLU', 'lce_HAM', 'lce_RF', 'lce_VAS', 'lce_BFSH', 'lce_TA'
        'F_GLU', 'F_HAM', 'F_RF', 'F_VAS', 'F_GAS', 'F_SOL'
        ]
    # control states
    cs_keys = [
        'ph_st', # leg in stance
        'ph_st_csw', # leg in stance ^ contra-leg in swing
        'ph_st_sw0', # leg in stance ^ initial swing
        'ph_sw', # leg in swing
        'ph_sw_flex_k', # leg in swing ^ flex knees
        'ph_sw_hold_k', # leg in swing ^ hold knee
        'ph_sw_stop_l', # leg in swing ^ stop leg
        'ph_sw_hold_l' # leg in swing ^ hold leg
        ]
    # brain commands
    bc_keys = [ 'theta_tgt', 'alpha_tgt', 'hip_tgt', 'knee_tgt', 'knee_sw_tgt',
        'alpha2RF', 'alpha0RF', 'alpha2HAM', 'alpha0HAM'
        'hip2HFL', 'hip0HFL', 'hip2GLU', 'hip0GLU',
        'knee2VAS', 'knee0VAS', 'knee2BFSH', 'knee0BFSH'
        ]
    # control parameters
    cp_keys = [
        'theta_tgt', 'c0', 'cv', 'alpha_delta',
        'knee_sw_tgt', 'knee_tgt',
        'HFL_3_PG', 'HFL_3_DG', 'HFL_4_G_CGLU', 'HFL_4_G_CHAM', 'HFL_6_LG',
        'HFL_6_VG', 'HFL_10_LG',
        'GLU_1_FG', 'GLU_3_PG', 'GLU_3_DG', 'GLU_4_G_CHFL', 'GLU_4_G_CRF',
        'GLU_6_LG', 'GLU_6_VG', 'GLU_10_LG',
        'HAM_2_FG', 'HAM_3_GLU', 'HAM_4_GLU', 'HAM_9_LG',
        'RF_1_FG', 'RF_8_VG',
        'VAS_1_FG', 'VAS_2_LG_BFSH', 'VAS_10_LG',
        'BFSH_2_LG', 'BFSH_2_loff', 'BFSH_7_VG_RF', 'BFSH_7_LG', 'BFSH_8_VG',
        'BFSH_9_G_HAM', 'BFSH_9_HAM0', 'BFSH_10_LG',
        'GAS_2_FG', 'GAS_9_G_HAM', 'GAS_9_HAM0',
        'SOL_1_FG',
        'TA_5_LG', 'TA_5_loff', 'TA_5_st_FG_SOL'
        ]

    m_map = dict(zip(m_keys, range(len(m_keys))))
    s_b_map = dict(zip(s_b_keys, range(len(s_b_keys))))
    s_l_map = dict(zip(s_l_keys, range(len(s_l_keys))))
    cs_map = dict(zip(cs_keys, range(len(cs_keys))))
    bc_map = dict(zip(bc_keys, range(len(bc_keys))))
    cp_map = dict(zip(cp_keys, range(len(cp_keys))))

    def __init__(self, TIMESTEP, control_mode=1, params = np.ones(len(cp_keys))):
        if self.DEBUG:
            print("===========================================")
            print("locomotion controller created in DEBUG mode")
            print("===========================================")

        self.dt = TIMESTEP
        self.t = 0
        self.n_step = 0

        self.control_mode = control_mode
        # 0: spinal control (no brain control)
        # 1: full control
        self.control_dimension = 2 # 2D or 3D

        if self.control_mode == 0:
            self.brain_control_on = 0
        elif self.control_mode == 1:
            self.brain_control_on = 1

        self.spinal_control_phase = {}
        # self.spinal_control_phase['r_leg'] = {}
        # self.spinal_control_phase['r_leg'] = {}
        self.in_contact = {}
        self.in_contact['r_leg'] = 1
        self.in_contact['l_leg'] = 0

        self.brain_command = {}

        spinal_control_phase_r = {}
        spinal_control_phase_r['ph_st'] = 1
        spinal_control_phase_r['ph_st_csw'] = 0
        spinal_control_phase_r['ph_st_sw0'] = 0
        spinal_control_phase_r['ph_st_st'] = 0
        spinal_control_phase_r['ph_sw'] = 0
        spinal_control_phase_r['ph_sw_flex_k'] = 0
        spinal_control_phase_r['ph_sw_hold_k'] = 0
        spinal_control_phase_r['ph_sw_stop_l'] = 0
        spinal_control_phase_r['ph_sw_hold_l'] = 0
        self.spinal_control_phase['r_leg'] = spinal_control_phase_r

        spinal_control_phase_l = {}
        spinal_control_phase_l['ph_st'] = 0
        spinal_control_phase_l['ph_st_csw'] = 0
        spinal_control_phase_l['ph_st_sw0'] = 0
        spinal_control_phase_l['ph_st_st'] = 0
        spinal_control_phase_l['ph_sw'] = 1
        spinal_control_phase_l['ph_sw_flex_k'] = 1
        spinal_control_phase_l['ph_sw_hold_k'] = 0
        spinal_control_phase_l['ph_sw_stop_l'] = 0
        spinal_control_phase_l['ph_sw_hold_l'] = 0
        self.spinal_control_phase['l_leg'] = spinal_control_phase_l

        self.stim = {}
        self.stim['r_leg'] = dict(zip(self.m_keys, 0.01*np.ones(len(self.m_keys))))
        self.stim['l_leg'] = dict(zip(self.m_keys, 0.01*np.ones(len(self.m_keys))))
        
        self._sensor_calibration()

        self.set_control_params(params)

    def _sensor_calibration(self):
        r_RF_h = -0.08; rho_RF = .5; lopt_RF = 0.08; phi0_RF_h = -10.*np.pi/180.; phi0_RF_k = -55.*np.pi/180.
        r_HAM_h = 0.08; rho_HAM = .5; lopt_HAM = 0.10; phi0_HAM_h = -30.*np.pi/180.; phi0_HAM_k = -0.*np.pi/180.
        r_HFL = -0.08; rho_HFL = .5; lopt_HFL = 0.11; phi0_HFL = -20.*np.pi/180.
        r_GLU = 0.08; rho_GLU = .5; lopt_GLU = 0.11; phi0_GLU = -60.*np.pi/180.
        r_VAS = 0.05; rho_VAS = .6; lopt_VAS = 0.08; phi0_VAS = -60.*np.pi/180.
        r_BFSH = -0.05; rho_BFSH = .6; lopt_BFSH = 0.12; phi0_BFSH = -30.*np.pi/180.

        self.sensor_gain = {}
        self.sensor_gain['alpha2RF'] = r_RF_h*rho_RF/lopt_RF
        self.sensor_gain['alpha0RF'] = 0.5*np.pi + phi0_RF_h - phi0_RF_k/2
        self.sensor_gain['alpha2HAM'] = r_HAM_h*rho_HAM/lopt_HAM
        self.sensor_gain['alpha0HAM'] = 0.5*np.pi + phi0_HAM_h - phi0_HAM_k/2
        self.sensor_gain['hip2HFL'] = r_HFL*rho_HFL/lopt_HFL
        self.sensor_gain['hip0HFL'] = phi0_HFL
        self.sensor_gain['hip2GLU'] = r_GLU*rho_GLU/lopt_GLU
        self.sensor_gain['hip0GLU'] = phi0_GLU
        self.sensor_gain['knee2VAS'] = r_VAS*rho_VAS/lopt_VAS
        self.sensor_gain['knee0VAS'] = phi0_VAS
        self.sensor_gain['knee2BFSH'] = r_BFSH*rho_BFSH/lopt_BFSH
        self.sensor_gain['knee0BFSH'] = phi0_BFSH

    def set_control_params(self, params):
        cp = {}
        cp_map = self.cp_map

        cp['theta_tgt'] = params[cp_map['cv']] *10*np.pi/180
        cp['c0'] = params[cp_map['c0']] *75*np.pi/180
        cp['cv'] = params[cp_map['cv']] *5*np.pi/180
        cp['alpha_delta'] = params[cp_map['alpha_delta']] *10*np.pi/180
        cp['knee_sw_tgt'] = -params[cp_map['knee_sw_tgt']] *50*np.pi/180
        cp['knee_tgt'] = (1-params[cp_map['knee_tgt']]) *5*np.pi/180

        cp['HFL_3_PG'] = params[cp_map['HFL_3_PG']] *1.0
        cp['HFL_3_DG'] = params[cp_map['HFL_3_DG']] *0.3
        cp['HFL_4_G_CGLU'] = params[cp_map['HFL_4_G_CGLU']] *0.1
        cp['HFL_4_G_CHAM'] = params[cp_map['HFL_4_G_CHAM']] *0.1
        cp['HFL_6_LG'] = params[cp_map['HFL_6_LG']] *1.0/np.abs(self.sensor_gain['alpha2RF'])
        cp['HFL_6_VG'] = params[cp_map['HFL_6_VG']] *.5
        cp['HFL_10_LG'] = params[cp_map['HFL_10_LG']] *.4
        cp['GLU_1_FG'] = params[cp_map['GLU_1_FG']] *1.0
        cp['GLU_3_PG'] = params[cp_map['GLU_3_PG']] *0.5
        cp['GLU_3_DG'] = params[cp_map['GLU_3_DG']] *0.1
        cp['GLU_4_G_CHFL'] = params[cp_map['GLU_4_G_CHFL']] *0.1
        cp['GLU_4_G_CRF'] = params[cp_map['GLU_4_G_CRF']] *0.1
        cp['GLU_6_LG'] = params[cp_map['GLU_6_LG']] *0.5/np.abs(self.sensor_gain['alpha2HAM'])
        cp['GLU_6_VG'] = params[cp_map['GLU_6_VG']] *0.5
        cp['GLU_10_LG'] = params[cp_map['GLU_10_LG']] *0.4
        cp['HAM_2_FG'] = params[cp_map['HAM_2_FG']] *1.0
        cp['HAM_3_GLU'] = params[cp_map['HAM_3_GLU']] *1.0
        cp['HAM_4_GLU'] = params[cp_map['HAM_4_GLU']] *1.0
        cp['HAM_9_LG'] = params[cp_map['HAM_9_LG']] *2.0/np.abs(self.sensor_gain['alpha2HAM'])
        cp['RF_1_FG'] = params[cp_map['RF_1_FG']] *0.1
        cp['RF_8_VG'] = params[cp_map['RF_8_VG']] *0.1/np.abs(self.sensor_gain['knee2BFSH'])
        cp['VAS_1_FG'] = params[cp_map['VAS_1_FG']] *1.2
        cp['VAS_2_LG_BFSH'] = params[cp_map['VAS_2_LG_BFSH']] *1.0
        cp['VAS_10_LG'] = params[cp_map['VAS_10_LG']] *0.3
        cp['BFSH_2_LG'] = params[cp_map['BFSH_2_LG']] *1.0
        cp['BFSH_2_loff'] = params[cp_map['BFSH_2_loff']] *1.2
        cp['BFSH_7_VG_RF'] = params[cp_map['BFSH_7_VG_RF']] *0.4/np.abs(self.sensor_gain['alpha2RF'])
        cp['BFSH_7_LG'] = params[cp_map['BFSH_7_LG']] *1.0/np.abs(self.sensor_gain['knee2BFSH'])
        cp['BFSH_8_VG'] = params[cp_map['BFSH_8_VG']] *3.0
        cp['BFSH_9_G_HAM'] = params[cp_map['BFSH_9_G_HAM']] *6.0
        cp['BFSH_9_HAM0'] = params[cp_map['BFSH_9_HAM0']] *0.6
        cp['BFSH_10_LG'] = params[cp_map['BFSH_10_LG']] *0.3
        cp['GAS_2_FG'] = params[cp_map['GAS_2_FG']] *1.2
        cp['GAS_9_G_HAM'] = params[cp_map['GAS_9_G_HAM']] *2.0
        cp['GAS_9_HAM0'] = params[cp_map['GAS_9_HAM0']] *6.0
        cp['SOL_1_FG'] = params[cp_map['SOL_1_FG']] *1.2
        cp['TA_5_LG'] = params[cp_map['TA_5_LG']] *1.1
        cp['TA_5_loff'] = params[cp_map['TA_5_loff']] *0.6
        cp['TA_5_st_FG_SOL'] = params[cp_map['TA_5_st_FG_SOL']] *0.4

        self.cp = cp

    def update(self, obs):
        self.t += self.dt
        sensor_data = self._update_sensor(obs)
        self.sensor_data = sensor_data

        if self.brain_control_on:
            # update self.brain_command
            self._brain_control(sensor_data)
        
        # updates self.stim
        self._spinal_control(sensor_data)

        # count step
        # !!!
        # if (not self.in_contact[s_leg] and sensor_data[s_leg]['contact_ipsi']) \
        #     and self.last_HS is not i_leg:
        #     self.n_step += 1
        #     self.last_HS = s_leg


        stim = np.array([self.stim['r_leg']['HFL'], self.stim['r_leg']['GLU'],
            self.stim['r_leg']['HAM'], self.stim['r_leg']['RF'],
            self.stim['r_leg']['VAS'], self.stim['r_leg']['BFSH'],
            self.stim['r_leg']['GAS'], self.stim['r_leg']['SOL'],
            self.stim['r_leg']['TA'],
            self.stim['l_leg']['HFL'], self.stim['l_leg']['GLU'],
            self.stim['l_leg']['HAM'], self.stim['l_leg']['RF'],
            self.stim['l_leg']['VAS'], self.stim['l_leg']['BFSH'],
            self.stim['l_leg']['GAS'], self.stim['l_leg']['SOL'],
            self.stim['l_leg']['TA']
            ])
        # todo: self._flaten(self.stim)
        return stim

    def _update_sensor(self, obs):
        sensor_data = {}

        sensor_data_body = {}
        sensor_data_body['theta'] = obs['roottheta']
        sensor_data_body['dx'] = obs['d_rootx']
        #sensor_data_body['dy'] = 0
        #sensor_data_body['dz'] = 0
        sensor_data_body['dtheta'] = obs['d_roottheta']
        sensor_data['body'] = sensor_data_body

        for s_l in ['r', 'l']:
            s_l_contra = 'l' if s_l is 'r' else 'r'
            s_leg = '{}_leg'.format(s_l)
            s_leg_contra = 'l_leg' if s_l is 'r' else 'r_leg'
            sensor_data_leg = {}
            sensor_data_leg['trunk_x'] = obs['{}_trunk_x'.format(s_l)]
            # sensor_data_leg['trunk_y'] = obs['{}_trunk_y'.format(s_l)]
            # sensor_data_leg['trunk_z'] = obs['{}_trunk_z'.format(s_l)]
            sensor_data_leg['trunk_dx'] = obs['{}_trunk_dx'.format(s_l)]
            # sensor_data_leg['trunk_dy'] = obs['{}_trunk_dy'.format(s_l)]
            # sensor_data_leg['trunk_dz'] = obs['{}_trunk_dz'.format(s_l)]
            sensor_data_leg['contact_ipsi'] = obs['{}_is_contact'.format(s_l)]
            sensor_data_leg['contact_contra'] = obs['{}_is_contact'.format(s_l_contra)]
            sensor_data_leg['load_ipsi'] = obs['{}_GRF_z'.format(s_l)]
            sensor_data_leg['load_contra'] = obs['{}_GRF_z'.format(s_l_contra)]
            sensor_data_leg['S_CHFL'] = self.stim[s_leg_contra]['HFL']
            sensor_data_leg['S_CGLU'] = self.stim[s_leg_contra]['GLU']
            sensor_data_leg['S_CHAM'] = self.stim[s_leg_contra]['HAM']
            sensor_data_leg['S_CRF'] = self.stim[s_leg_contra]['RF']
            sensor_data_leg['lce_HFL'] = obs['{}_HFL_l_ce'.format(s_l)]
            sensor_data_leg['lce_GLU'] = obs['{}_GLU_l_ce'.format(s_l)]
            sensor_data_leg['F_GLU'] = obs['{}_GLU_F_mtu'.format(s_l)]
            sensor_data_leg['vce_HAM'] = obs['{}_HAM_v_ce'.format(s_l)]
            sensor_data_leg['lce_HAM'] = obs['{}_HAM_l_ce'.format(s_l)]
            sensor_data_leg['F_HAM'] = obs['{}_HAM_F_mtu'.format(s_l)]
            sensor_data_leg['lce_RF'] = obs['{}_RF_l_ce'.format(s_l)]
            sensor_data_leg['vce_RF'] = obs['{}_RF_v_ce'.format(s_l)]
            sensor_data_leg['F_RF'] = obs['{}_RF_F_mtu'.format(s_l)]
            sensor_data_leg['vce_VAS'] = obs['{}_VAS_v_ce'.format(s_l)]
            sensor_data_leg['lce_VAS'] = obs['{}_VAS_l_ce'.format(s_l)]
            sensor_data_leg['F_VAS'] = obs['{}_VAS_F_mtu'.format(s_l)]
            sensor_data_leg['vce_BFSH'] = obs['{}_BFSH_v_ce'.format(s_l)]
            sensor_data_leg['lce_BFSH'] = obs['{}_BFSH_l_ce'.format(s_l)]
            sensor_data_leg['F_GAS'] = obs['{}_GAS_F_mtu'.format(s_l)]
            sensor_data_leg['F_SOL'] = obs['{}_SOL_F_mtu'.format(s_l)]
            sensor_data_leg['lce_TA'] = obs['{}_TA_l_ce'.format(s_l)]
            sensor_data[s_leg] = sensor_data_leg

        return sensor_data

    def _brain_control(self, sensor_data=0):
        s_r = sensor_data['r_leg']
        s_l = sensor_data['l_leg']
        s_b = sensor_data['body']
        cp = self.cp

        alpha2HAM = self.sensor_gain['alpha2HAM']
        alpha0HAM = self.sensor_gain['alpha0HAM']

        # (todo) define angles based on PHI0 and S_PHI
        # self.brain_command[bc_map["theta_tgt"]] = S_THETA*10*np.pi/180 - THETA0

        self.brain_command['theta_tgt'] = cp['theta_tgt']

        self.brain_command['r_leg'] = {}
        self.brain_command['l_leg'] = {}
        for s_leg in ['r_leg', 'l_leg']:
            # !!! todo: calculate alpha_tgt_global
            # based on hind limb or contra/ipsi-limblce_HFL
            self.brain_command[s_leg]['alpha_tgt_global'] = cp['c0'] - cp['cv']*s_b['dx']
            # alpha = 0.5*np.pi + hip - 0.5*knee
            self.brain_command[s_leg]['alpha_tgt'] = self.brain_command[s_leg]['alpha_tgt_global'] \
                                            - s_b['theta']
            self.brain_command[s_leg]['alpha_delta'] = cp['alpha_delta']
            self.brain_command[s_leg]['knee_sw_tgt'] = cp['knee_sw_tgt']
            self.brain_command[s_leg]['knee_tgt'] = cp['knee_tgt']
            self.brain_command[s_leg]['hip_tgt'] = -0.5*np.pi + self.brain_command[s_leg]['alpha_tgt'] + 0.5*self.brain_command[s_leg]['knee_tgt']

        # select which leg to swing
        self.brain_command['r_leg']['swing_init'] = 0
        self.brain_command['l_leg']['swing_init'] = 0
        if sensor_data['r_leg']['contact_ipsi'] and sensor_data['l_leg']['contact_ipsi']:
            r_alpha = self._lce2phi(s_r['lce_HAM'], alpha2HAM, alpha0HAM)
            r_delta_alpha = r_alpha - self.brain_command['r_leg']['alpha_tgt']
            l_alpha = self._lce2phi(s_l['lce_HAM'], alpha2HAM, alpha0HAM)
            l_delta_alpha = l_alpha - self.brain_command['l_leg']['alpha_tgt']
            if r_delta_alpha > l_delta_alpha:
                self.brain_command['r_leg']['swing_init'] = 1
            else:
                self.brain_command['l_leg']['swing_init'] = 1
    
    def _spinal_control(self, sensor_data):
        for s_leg in ['r_leg', 'l_leg']:
            self._update_spinal_control_phase(s_leg, sensor_data)
            self.stim[s_leg] = self.spinal_control_leg(s_leg, sensor_data)

    def _update_spinal_control_phase(self, s_leg, sensor_data):
        s_l = sensor_data[s_leg]

        alpha_tgt = self.brain_command[s_leg]['alpha_tgt']
        alpha_delta = self.brain_command[s_leg]['alpha_delta']
        knee_sw_tgt = self.brain_command[s_leg]['knee_sw_tgt']

        alpha2HAM = self.sensor_gain['alpha2HAM']
        alpha0HAM = self.sensor_gain['alpha0HAM']
        knee2BFSH = self.sensor_gain['knee2BFSH']
        knee0BFSH = self.sensor_gain['knee0BFSH']

        # when foot touches ground
        if not self.in_contact[s_leg] and s_l['contact_ipsi']:
            # initiate stance control
            self.spinal_control_phase[s_leg]['ph_st'] = 1
            # swing control off
            self.spinal_control_phase[s_leg]['ph_sw'] = 0
            self.spinal_control_phase[s_leg]['ph_sw_flex_k'] = 0
            self.spinal_control_phase[s_leg]['ph_sw_hold_k'] = 0
            self.spinal_control_phase[s_leg]['ph_sw_stop_l'] = 0
            self.spinal_control_phase[s_leg]['ph_sw_hold_l'] = 0

        # during stance control
        if self.spinal_control_phase[s_leg]['ph_st']:
            # contra-leg in swing (single stance phase)
            self.spinal_control_phase[s_leg]['ph_st_csw'] = not s_l['contact_contra']
            # initiate swing
            self.spinal_control_phase[s_leg]['ph_st_sw0'] = self.brain_command[s_leg]['swing_init']
            # do not initiate swing
            self.spinal_control_phase[s_leg]['ph_st_st'] = not self.spinal_control_phase[s_leg]['ph_st_sw0']

        # when foot loses contact
        if self.in_contact[s_leg] and not s_l['contact_ipsi']:
            # stance control off
            self.spinal_control_phase[s_leg]['ph_st'] = 0
            self.spinal_control_phase[s_leg]['ph_st_csw'] = 0
            self.spinal_control_phase[s_leg]['ph_st_sw0'] = 0
            self.spinal_control_phase[s_leg]['ph_st_st'] = 0
            # initiate swing control
            self.spinal_control_phase[s_leg]['ph_sw'] = 1
            # flex knee
            self.spinal_control_phase[s_leg]['ph_sw_flex_k'] = 1

        # during swing control
        if self.spinal_control_phase[s_leg]['ph_sw']:
            if self.spinal_control_phase[s_leg]['ph_sw_flex_k']:
                # !!! check: VASorBFSH
                if self._lce2phi(s_l['lce_BFSH'], knee2BFSH, knee0BFSH) < knee_sw_tgt: # knee flexed
                    self.spinal_control_phase[s_leg]['ph_sw_flex_k'] = 0
                    # hold knee
                    self.spinal_control_phase[s_leg]['ph_sw_hold_k'] = 1
            else:
                alpha = self._lce2phi(s_l['lce_HAM'], alpha2HAM, alpha0HAM)
                if self.spinal_control_phase[s_leg]['ph_sw_hold_k']:
                    if alpha < alpha_tgt: # leg swung enough
                        self.spinal_control_phase[s_leg]['ph_sw_hold_k'] = 0
                if alpha < alpha_tgt + alpha_delta: # leg swung enough
                    # stop leg
                    self.spinal_control_phase[s_leg]['ph_sw_stop_l'] = 1
                if self.spinal_control_phase[s_leg]['ph_sw_stop_l'] \
                    and s_l['vce_HAM'] < 0: # leg started to retract
                    # hold leg
                    self.spinal_control_phase[s_leg]['ph_sw_hold_l'] = 1

        self.in_contact[s_leg] = s_l['contact_ipsi']

    def spinal_control_leg(self, s_leg, sensor_data):
        s_l = sensor_data[s_leg]
        s_b = sensor_data['body']
        cp = self.cp

        ph_st = self.spinal_control_phase[s_leg]['ph_st']
        ph_st_csw = self.spinal_control_phase[s_leg]['ph_st_csw']
        ph_st_sw0 = self.spinal_control_phase[s_leg]['ph_st_sw0']
        ph_st_st = self.spinal_control_phase[s_leg]['ph_st_st']
        ph_sw = self.spinal_control_phase[s_leg]['ph_sw']
        ph_sw_flex_k = self.spinal_control_phase[s_leg]['ph_sw_flex_k']
        ph_sw_hold_k = self.spinal_control_phase[s_leg]['ph_sw_hold_k']
        ph_sw_stop_l = self.spinal_control_phase[s_leg]['ph_sw_stop_l']
        ph_sw_hold_l = self.spinal_control_phase[s_leg]['ph_sw_hold_l']

        theta_tgt = self.brain_command['theta_tgt']
        alpha_tgt = self.brain_command[s_leg]['alpha_tgt']
        alpha_delta = self.brain_command[s_leg]['alpha_delta']
        hip_tgt = self.brain_command[s_leg]['hip_tgt']
        knee_tgt = self.brain_command[s_leg]['knee_tgt']
        knee_sw_tgt = self.brain_command[s_leg]['knee_sw_tgt']

        alpha2RF = self.sensor_gain['alpha2RF']
        alpha0RF = self.sensor_gain['alpha0RF']
        alpha2HAM = self.sensor_gain['alpha2HAM']
        alpha0HAM = self.sensor_gain['alpha0HAM']
        hip2HFL = self.sensor_gain['hip2HFL']
        hip0HFL = self.sensor_gain['hip0HFL']
        hip2GLU = self.sensor_gain['hip2GLU']
        hip0GLU = self.sensor_gain['hip0GLU']
        knee2VAS = self.sensor_gain['knee2VAS']
        knee0VAS = self.sensor_gain['knee0VAS']
        knee2BFSH = self.sensor_gain['knee2BFSH']
        knee0BFSH = self.sensor_gain['knee0BFSH']

        stim = {}
        pre_stim = 0.01

        if self.control_dimension == 3:
            S_HAB = 1 #!!!
            S_HAD = 1 #!!!

        S_HFL_3 = ph_st*s_l['load_ipsi']*self._FB_PD(s_b['theta'], s_b['dtheta'],
            -cp['HFL_3_PG'], -cp['HFL_3_DG'], theta_tgt)
        S_HFL_4 = ph_st_csw*(self._FB(s_l['S_CGLU'], cp['HFL_4_G_CGLU'])
            + self._FB(s_l['S_CHAM'], cp['HFL_4_G_CHAM']))
        S_HFL_6 = (ph_st_sw0*s_l['load_contra'] + ph_sw) \
            *( self._FB(s_l['lce_RF'], cp['HFL_6_LG'],
                self._phi2lce(alpha_tgt, alpha2RF, alpha0RF))
                -self._FB(s_l['vce_RF'], cp['HFL_6_VG']) )
        S_HFL_10 = ph_sw_hold_l*self._FB(s_l['lce_HFL'], cp['HFL_10_LG'],
            self._phi2lce(hip_tgt, hip2HFL, hip0HFL))
        stim['HFL'] = pre_stim + S_HFL_3 + S_HFL_4 + S_HFL_6 + S_HFL_10

        S_GLU_1 = (ph_st_st + ph_st_sw0*(1-s_l['load_contra'])) \
            *self._FB(s_l['F_GLU'], cp['GLU_1_FG'])
        S_GLU_3 = ph_st*s_l['load_ipsi']*self._FB_PD(s_b['theta'], s_b['dtheta'],
            cp['GLU_3_PG'], cp['GLU_3_DG'], theta_tgt)
        S_GLU_4 = ph_st_csw*(self._FB(s_l['S_CHFL'], cp['GLU_4_G_CHFL']) \
            + self._FB(s_l['S_CRF'], cp['GLU_4_G_CRF']))
        S_GLU_6 = (ph_st_sw0*s_l['load_contra'] + ph_sw) \
            *( self._FB(s_l['lce_HAM'], cp['GLU_6_LG'],
                self._phi2lce(alpha_tgt + alpha_delta, alpha2HAM, alpha0HAM))
                -self._FB(s_l['vce_HAM'], cp['GLU_6_VG']) )
        S_GLU_10 = ph_sw_hold_l*self._FB(s_l['lce_GLU'], cp['GLU_10_LG'],
            self._phi2lce(hip_tgt, hip2GLU, hip0GLU))
        stim['GLU'] = pre_stim + S_GLU_1 + S_GLU_3 + S_GLU_4 + S_GLU_6 + S_GLU_10

        S_HAM_2 = (ph_st_st + ph_st_sw0*(1-s_l['load_contra'])) \
            *self._FB(s_l['F_HAM'], cp['HAM_2_FG'])
        S_HAM_3 = cp['HAM_3_GLU']*S_GLU_3
        S_HAM_4 = cp['HAM_4_GLU']*S_GLU_4
        S_HAM_9 = ph_sw_stop_l*self._FB(s_l['lce_HAM'], cp['HAM_9_LG'],
            self._phi2lce(alpha_tgt + alpha_delta, alpha2HAM, alpha0HAM))
        stim['HAM'] = pre_stim + S_HAM_2 + S_HAM_3 + S_HAM_4 + S_HAM_9

        S_RF_1 = (ph_st_st + ph_st_sw0*(1-s_l['load_contra'])) \
            *self._FB(s_l['F_RF'], cp['RF_1_FG'])
        S_RF_8 = ph_sw_hold_k*self._FB(s_l['vce_VAS'], cp['RF_8_VG'])
        stim['RF'] = pre_stim + S_RF_1 + S_RF_8

        S_VAS_1 = (ph_st_st + ph_st_sw0*(1-s_l['load_contra'])) \
            *self._FB(s_l['F_VAS'], cp['VAS_1_FG'])
        S_VAS_2 = -(ph_st_st + ph_st_sw0*(1-s_l['load_contra'])) \
            *self._FB(s_l['lce_BFSH'], cp['VAS_2_LG_BFSH'], cp['BFSH_2_loff'])
        S_VAS_10 = ph_sw_hold_l*self._FB(s_l['lce_VAS'], cp['VAS_10_LG'],
            self._phi2lce(knee_tgt, knee2VAS, knee0VAS) )
        stim['VAS'] = pre_stim + S_VAS_1 + S_VAS_2 + S_VAS_10

        S_BFSH_2 = (ph_st_st + ph_st_sw0*(1-s_l['load_contra'])) \
            *self._FB(s_l['lce_BFSH'], cp['BFSH_2_LG'], cp['BFSH_2_loff'])
        S_BFSH_7 = (ph_st_sw0*(s_l['load_contra']) + ph_sw_flex_k) \
            * ( self._FB(s_l['vce_RF'], cp['BFSH_7_VG_RF'])
                + self._FB(s_l['lce_BFSH'], cp['BFSH_7_LG'],
                    self._phi2lce(knee_sw_tgt, knee2BFSH, knee0BFSH) ) )
        S_BFSH_8 = ph_sw_hold_k \
            *self._FB(s_l['vce_BFSH'], cp['BFSH_8_VG']) \
            *self._FB(s_l['lce_RF'], 1/alpha2RF,
                self._phi2lce(alpha_tgt, alpha2RF, alpha0RF) )
        S_BFSH_9 = self._FB(S_HAM_9, cp['BFSH_9_G_HAM'], cp['BFSH_9_HAM0'])
        S_BFSH_10 = ph_sw_hold_l*self._FB(s_l['lce_BFSH'], cp['BFSH_10_LG'],
            self._phi2lce(knee_tgt, knee2BFSH, knee0BFSH) )
        stim['BFSH'] = pre_stim + S_BFSH_2 + S_BFSH_7 + S_BFSH_8 + S_BFSH_9 + S_BFSH_10

        S_GAS_2 = ph_st*self._FB(s_l['F_GAS'], cp['GAS_2_FG'])
        S_GAS_9 = self._FB(S_HAM_9, cp['GAS_9_G_HAM'], cp['GAS_9_HAM0'])
        stim['GAS'] = pre_stim + S_GAS_2 + S_GAS_9

        S_SOL_1 = ph_st*self._FB(s_l['F_SOL'], cp['SOL_1_FG'])
        stim['SOL'] = pre_stim + S_SOL_1

        S_TA_5 = self._FB(s_l['lce_TA'], cp['TA_5_LG'], cp['TA_5_loff'])
        S_TA_5_st = -ph_st*self._FB(s_l['F_SOL'], cp['TA_5_st_FG_SOL'])
        stim['TA'] = pre_stim + S_TA_5 + S_TA_5_st

        for muscle in stim:
            stim[muscle] = np.clip(stim[muscle], 0.01, 1.0)

        return stim

    # basic feedback circuit
    def _FB(self, data, G, offset=0):
        return G*np.maximum(data-offset, 0)

    # PD feedback circuit
    def _FB_PD(self, data, ddata, PG, DG, offset):
        return np.maximum(PG*(data-offset) + DG*ddata, 0)

    def _phi2lce(self, phi, C, phi0):
        # larger phi results in shorter l_mtu in case of extensor muscles (r > 0)
        lce = 1 - C*(phi - phi0)
        return lce

    def _lce2phi(self, lce, C, phi0):
        # larger phi results in shorter l_mtu in case of extensor muscles (r > 0)
        phi = -(lce - 1)/C + phi0
        return phi