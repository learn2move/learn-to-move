from v_tgt_field import VTgtField
import sys
import numpy as np
import matplotlib.pyplot as plt
# --------------------------------------------------------------------

...

# --------------------------------------------------------------------
#rng_xy = np.array([[-30, 30], [-30, 30]])
rng_xy = np.array([[-10, 10], [-5, 5]])
vtgt_obj = VTgtField(rng_xy)
plt.axes([.025, .025, .95, .95])
X = vtgt_obj.grid[0]
Y = vtgt_obj.grid[1]
U = vtgt_obj.vect[0]
V = vtgt_obj.vect[1]
R = np.sqrt(U**2 + V**2)
plt.quiver(X, Y, U, V, R)
plt.axis('equal')
plt.show()

p_sink = [3.8,2.5] # [x, y]
d_sink = np.linalg.norm(p_sink)
v_amp_rng = [1.0, 2.0]
vtgt_obj.create_vtgt_sink(p_sink, d_sink, v_amp_rng, v_phase0=np.pi)

plt.axes([.025, .025, .95, .95])
X = vtgt_obj.grid[0]
Y = vtgt_obj.grid[1]
U = vtgt_obj.vect[0]
V = vtgt_obj.vect[1]
R = np.sqrt(U**2 + V**2)
plt.quiver(X, Y, U, V, R)
plt.axis('equal')
plt.show()

import pdb; pdb.set_trace()


pose = [0, 0, 0] # [x, y, theta]
vtgt_obj.query_vtgt(pose)


# --------------------------------------------------------------------