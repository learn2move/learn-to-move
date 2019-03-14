from song_2018 import MusculoTendonJoint as MTJ
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


"""
### test muscle modules
from song_2018 import fn_inv_f_vce0
from song_2018 import fn_inv_f_vce0_2 # check where this equationo came from
from song_2018 import fn_f_lce0
f_v = np.linspace(.1, 1.6, 100)
v_ce = np.empty([100,1])
v_ce_2 = np.empty([100,1])
l_ce = np.linspace(.6, 1.4, 100)
f_l = np.empty([100,1])
for i in range(100):
    v_ce[i] = fn_inv_f_vce0(f_v[i], MTJ.K, MTJ.N)
    v_ce_2[i] = fn_inv_f_vce0_2(f_v[i], MTJ.K, MTJ.N)
    f_l[i] = fn_f_lce0(l_ce[i], MTJ.W, MTJ.C)

plt.figure()
plt.subplot(3,1,1)
plt.plot(v_ce, f_v)
plt.hold(True)
plt.plot(v_ce_2, f_v)
plt.subplot(3,1,2)
plt.plot(l_ce, f_l)
"""


muscle_dt = 0.008
n_update_muscle = 4
#def _get_mtj(self, mtu_par0, phi0, phi_ref, name, r1):
def _get_mtj(mtu_par0, phi0, phi_ref, name, r1):
        mtu_par = dict(mtu_par0)
        mtu_par.update({
            'name': name,
            'r1': r1,
            'phi1_ref': phi_ref,
            'phi1_0': phi0,
            'A': 0.01
            })
        
        return MTJ(muscle_dt, mtu_par, n_update=n_update_muscle)

def test_static(mscl, act, phi):
    t = 0
    t_end = .4
    n_dt = int(t_end/muscle_dt)
    v_t = np.empty([n_dt,1])
    trq = np.empty([n_dt,1])
    F_mtu_0 = np.empty([n_dt,1])
    l_ce_0 = np.empty([n_dt,1])
    v_ce_0 = np.empty([n_dt,1])
    for i in range(n_dt):
        t += muscle_dt
        v_t[i] = t
        m_stim = act
        mscl.update(m_stim, phi)
        name_0, F_mtu_0[i], l_ce_0[i], v_ce_0[i] = mscl.getStates()
        trq[i] = mscl.getTorque()
    return (v_t, trq, F_mtu_0, l_ce_0, v_ce_0)

def test_len(mscl, act, phi_range, hz=[1, 1]):
    t = 0
    t_end = 5
    n_dt = int(t_end/muscle_dt)
    v_t = np.empty([n_dt,1])
    trq = np.empty([n_dt,1])
    F_mtu_0 = np.empty([n_dt,1])
    l_ce_0 = np.empty([n_dt,1])
    v_ce_0 = np.empty([n_dt,1])
    for i in range(n_dt):
        t += muscle_dt
        v_t[i] = t
        phi0 = np.mean(phi_range)
        phi_amp = .5*(phi_range[1] - phi_range[0])
        phi = phi0 + phi_amp*signal.chirp(t, hz[0], t_end, hz[1], phi=90)
        m_stim = act
        mscl.update(m_stim, phi)
        name_0, F_mtu_0[i], l_ce_0[i], v_ce_0[i] = mscl.getStates()
        trq[i] = mscl.getTorque()
    return (v_t, trq, F_mtu_0, l_ce_0, v_ce_0)

ctrl_range = [-1.0, 1.0]
gear = 200
phi_range = [-45*np.pi/180, 45*np.pi/180]

trq_range = [gear*ctrl_range[0], gear*ctrl_range[1]]

del_phi = phi_range[1] - phi_range[0]
i_m = 1
r0 = np.sign(trq_range[i_m])*0.04/del_phi
F_max = np.abs(trq_range[i_m]/r0)
phi_ref = np.mean(phi_range) + np.sign(trq_range[i_m])*del_phi*.2

print("r0:{}".format(r0))
print("F_max:{}".format(F_max))

mtu_par0 = {
            'F_max': F_max,
            'l_opt': 0.1,
            'v_max': 12,
            'l_slack': 0.4,
            'rho': 1,
        }
phi0 = 0

plt.figure()

muscles = {}
muscles['m0'] = _get_mtj(mtu_par0, phi0=phi0, phi_ref=phi_ref, name='m0', r1=r0)

act = 1
(v_t, trq, F_mtu_0, l_ce_0, v_ce_0) = test_static(muscles['m0'], act, phi_range[0])

plt.subplot(4,5,1)
plt.plot(v_t, trq)
plt.ylim((trq_range[0]*1.5, trq_range[1]*1.5))
plt.subplot(4,5,6)
plt.plot(v_t, F_mtu_0)
plt.ylim((0, 1.5))
plt.subplot(4,5,11)
plt.plot(v_t, l_ce_0)
plt.ylim((.6, 1.3))
plt.subplot(4,5,16)
plt.plot(v_t, v_ce_0)
plt.ylim((-1.1, 1.1))
print("F_mtu_0: {} l_ce_0:{}".format(F_mtu_0[-1],l_ce_0[-1]))

muscles['m0'].reset()
act = 1
(v_t, trq, F_mtu_0, l_ce_0, v_ce_0) = test_static(muscles['m0'], act, phi_range[1])

plt.subplot(4,5,2)
plt.plot(v_t, trq)
plt.ylim((trq_range[0]*1.5, trq_range[1]*1.5))
plt.subplot(4,5,7)
plt.plot(v_t, F_mtu_0)
plt.ylim((0, 1.5))
plt.subplot(4,5,12)
plt.plot(v_t, l_ce_0)
plt.ylim((.6, 1.3))
plt.subplot(4,5,17)
plt.plot(v_t, v_ce_0)
plt.ylim((-1.1, 1.1))
print("F_mtu_0: {} l_ce_0:{}".format(F_mtu_0[-1],l_ce_0[-1]))

muscles['m0'].reset()
act = 0
(v_t, trq, F_mtu_0, l_ce_0, v_ce_0) = test_static(muscles['m0'], act, phi_range[0])

plt.subplot(4,5,3)
plt.plot(v_t, trq)
plt.ylim((trq_range[0]*1.5, trq_range[1]*1.5))
plt.subplot(4,5,8)
plt.plot(v_t, F_mtu_0)
plt.ylim((0, 1.5))
plt.subplot(4,5,13)
plt.plot(v_t, l_ce_0)
plt.ylim((.6, 1.3))
plt.subplot(4,5,18)
plt.plot(v_t, v_ce_0)
plt.ylim((-1.1, 1.1))
print("F_mtu_0: {} l_ce_0:{}".format(F_mtu_0[-1],l_ce_0[-1]))

muscles['m0'].reset()
act = 0
(v_t, trq, F_mtu_0, l_ce_0, v_ce_0) = test_static(muscles['m0'], act, np.mean(phi_range))

plt.subplot(4,5,4)
plt.plot(v_t, trq)
plt.ylim((trq_range[0]*1.5, trq_range[1]*1.5))
plt.subplot(4,5,9)
plt.plot(v_t, F_mtu_0)
plt.ylim((0, 1.5))
plt.subplot(4,5,14)
plt.plot(v_t, l_ce_0)
plt.ylim((.6, 1.3))
plt.subplot(4,5,19)
plt.plot(v_t, v_ce_0)
plt.ylim((-1.1, 1.1))
print("F_mtu_0: {} l_ce_0:{}".format(F_mtu_0[-1],l_ce_0[-1]))

muscles['m0'].reset()
act = 0
(v_t, trq, F_mtu_0, l_ce_0, v_ce_0) = test_static(muscles['m0'], act, phi_range[1])

plt.subplot(4,5,5)
plt.plot(v_t, trq)
plt.ylim((trq_range[0]*1.5, trq_range[1]*1.5))
plt.subplot(4,5,10)
plt.plot(v_t, F_mtu_0)
plt.ylim((0, 1.5))
plt.subplot(4,5,15)
plt.plot(v_t, l_ce_0)
plt.ylim((.6, 1.3))
plt.subplot(4,5,20)
plt.plot(v_t, v_ce_0)
plt.ylim((-1.1, 1.1))
print("F_mtu_0: {} l_ce_0:{}".format(F_mtu_0[-1],l_ce_0[-1]))

muscles['m0'].reset()
hz = [1, 5]
act = 1
(v_t, trq, F_mtu_0, l_ce_0, v_ce_0) = test_len(muscles['m0'], act, phi_range, hz)

plt.figure()
plt.subplot(411)
plt.plot(v_t, trq)
plt.ylim((trq_range[0]*1.5, trq_range[1]*1.5))
plt.subplot(412)
plt.plot(v_t, F_mtu_0)
plt.ylim((0, 1.5))
plt.subplot(413)
plt.plot(v_t, l_ce_0)
plt.ylim((.6, 1.3))
plt.subplot(414)
plt.plot(v_t, v_ce_0)
plt.ylim((-1.1, 1.1))


plt.show()

