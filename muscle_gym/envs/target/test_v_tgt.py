from v_tgt_field import VTgtField
import sys
import numpy as np
import matplotlib.pyplot as plt
# --------------------------------------------------------------------

#...

# --------------------------------------------------------------------
#rng_xy = np.array([[-30, 30], [-30, 30]])
rng_xy = np.array([[-10, 10], [-5, 5]])
vtgt_obj = VTgtField(rng_xy)

plt.figure(0)
plt.axes([.025, .025, .95, .95])
X = vtgt_obj.map[0]
Y = vtgt_obj.map[1]
U = vtgt_obj.vtgt[0]
V = vtgt_obj.vtgt[1]
R = np.sqrt(U**2 + V**2)
plt.quiver(X, Y, U, V, R)
plt.axis('equal')

p_sink = np.array([3.8,2.5]) # [x, y]
d_sink = np.linalg.norm(p_sink)
v_amp_rng = np.array([1.0, 2.0])
vtgt_obj.create_vtgt_sink(p_sink, d_sink, v_amp_rng, v_phase0=np.pi)

plt.figure(1)
plt.axes([.025, .025, .95, .95])
X = vtgt_obj.map[0]
Y = vtgt_obj.map[1]
U = vtgt_obj.vtgt[0]
V = vtgt_obj.vtgt[1]
R = np.sqrt(U**2 + V**2)
plt.quiver(X, Y, U, V, R)
plt.axis('equal')


pose = [3.8, 2.5, 30*np.pi/180] # [x, y, theta]
vtgt = vtgt_obj.get_vtgt(pose[0:2])

print('vtgt: {}'.format(vtgt))

vtgt_field_local = vtgt_obj.get_vtgt_field_local(pose)

plt.figure(2)
plt.axes([.025, .025, .95, .95])
X, Y = np.mgrid[-20:21, -10:11]/2
U = vtgt_field_local[0]
V = vtgt_field_local[1]
R = np.sqrt(U**2 + V**2)
plt.quiver(X, Y, U, V, R)
plt.axis('equal')
plt.show()


import pdb; pdb.set_trace()

# --------------------------------------------------------------------