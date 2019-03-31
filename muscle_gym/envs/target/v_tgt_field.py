# Author(s): Seungmoon Song <seungmoon.song@gmail.com>
"""
...
"""

from __future__ import division # '/' always means non-truncating division
import numpy as np


class VTgtField(object):
    res = 2

# -----------------------------------------------------------------------------------------------------------------
    def __init__(self, rng_xy=np.array([[-10, 10], [-10, 10]])):
        # origins and vectors
        self.grid = self._generate_grid(rng_xy, self.res)
        self.vect = -1*self.grid

# -----------------------------------------------------------------------------------------------------------------
    def __del__(self):
        nn = "empty"

# -----------------------------------------------------------------------------------------------------------------
    def create_vtgt_sink(self, p_sink, d_sink, v_amp_rng, v_phase0=np.random.uniform(-np.pi, np.pi)):
        # set vector orientations
        rng_xy = np.array([ -p_sink[0] + np.array([self.grid[0,0,0], self.grid[0,-1,0]]), 
                            -p_sink[1] + np.array([self.grid[1,0,0], self.grid[1,0,-1]])])
        self.vect = -self._generate_grid(rng_xy, self.res)

        # set vector amplitudes
        self._set_sink_vect_amp(p_sink, d_sink, v_amp_rng, v_phase0)

# -----------------------------------------------------------------------------------------------------------------
    def query_vtgt(self, pose):
        return

# -----------------------------------------------------------------------------------------------------------------
    def query_v_global(self, pose):
        return

# -----------------------------------------------------------------------------------------------------------------
    def _generate_grid(self, rng_xy=np.array([[-10, 10], [-10, 10]]), res=.5):        
        xo = .5*(rng_xy[0,0]+rng_xy[0,1])
        x_del = (rng_xy[0,1]-xo)*res
        yo = .5*(rng_xy[1,0]+rng_xy[1,1])
        y_del = (rng_xy[1,1]-yo)*res
        grid = np.mgrid[-x_del:x_del+1, -y_del:y_del+1]/res
        grid[0] = grid[0] + xo
        grid[1] = grid[1] + yo
        return grid

# -----------------------------------------------------------------------------------------------------------------
    def _set_sink_vect_amp(self, p_sink, d_sink, v_amp_rng, v_phase0, d_dec = 1):
        # d_dec: start to decelerate within d_dec of sink

        for i_x, x in enumerate(self.grid[0,:,0]):
            for i_y, y in enumerate(self.grid[1,0,:]):
                d = np.linalg.norm([ x-p_sink[0], y-p_sink[1] ])
                if d > d_sink + d_dec:
                    v_amp = v_amp_rng[1]
                elif d > d_dec:
                    v_phase = v_phase0 + d/d_sink*2*np.pi
                    v_amp = .5*np.diff(v_amp_rng)*np.sin(v_phase) + np.mean(v_amp_rng)
                else:
                    v_phase = v_phase0 + d_dec/d_sink*2*np.pi
                    v_amp0 = .5*np.diff(v_amp_rng)*np.sin(v_phase) + np.mean(v_amp_rng)
                    v_amp = d*v_amp0

                #import pdb; pdb.set_trace()

                self.vect[0,i_x,i_y] = v_amp*self.vect[0,i_x,i_y]/np.linalg.norm(self.vect[:,i_x,i_y])
                self.vect[1,i_x,i_y] = v_amp*self.vect[1,i_x,i_y]/np.linalg.norm(self.vect[:,i_x,i_y])

                #import pdb; pdb.set_trace()


        
