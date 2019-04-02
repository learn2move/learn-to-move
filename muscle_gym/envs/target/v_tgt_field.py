# Author(s): Seungmoon Song <seungmoon.song@gmail.com>
"""
...
"""

from __future__ import division # '/' always means non-truncating division
import numpy as np
from scipy import interpolate


class VTgtField(object):
    res_map = 2
    res_get = 2
    rng_get = np.array([[-10, 10], [-5, 5]])

# -----------------------------------------------------------------------------------------------------------------
    def __init__(self, rng_xy=np.array([[-10, 10], [-10, 10]])):
        # map coordinate and vtgt
        self.create_map(rng_xy)
        self.vtgt = -1*self.map

# -----------------------------------------------------------------------------------------------------------------
    def __del__(self):
        nn = "empty"

# -----------------------------------------------------------------------------------------------------------------
    def create_map(self, rng_xy):
        self.map_rng_xy = rng_xy
        self.map = self._generate_grid(rng_xy, self.res_map)

# -----------------------------------------------------------------------------------------------------------------
    def create_vtgt_sink(self, p_sink, d_sink, v_amp_rng, v_phase0=np.random.uniform(-np.pi, np.pi)):
        # set vtgt orientations
        rng_xy = (-p_sink + self.map_rng_xy.T).T
        self.vtgt = -self._generate_grid(rng_xy, self.res_map)

        # set vtgt amplitudes
        self._set_sink_vtgt_amp(p_sink, d_sink, v_amp_rng, v_phase0)

        self.vtgt_interp_x = interpolate.interp2d(self.map[0,:,0], self.map[1,0,:], self.vtgt[0].T, kind='linear')
        self.vtgt_interp_y = interpolate.interp2d(self.map[0,:,0], self.map[1,0,:], self.vtgt[1].T, kind='linear')

# -----------------------------------------------------------------------------------------------------------------
    def get_vtgt_field_local(self, pose):
        xy = pose[0:2]
        th = pose[2]

        # create query map
        get_rng_xy0 = (xy + self.rng_get.T).T
        get_map0 = self._generate_grid(get_rng_xy0, self.res_get)
        get_map_x = np.cos(th)*get_map0[0,:,:] - np.sin(th)*get_map0[1,:,:]
        get_map_y = np.sin(th)*get_map0[0,:,:] + np.cos(th)*get_map0[1,:,:]
-> rotate around center
        # get vtgt
        vtgt_x0 = np.reshape(np.array([self.vtgt_interp_x(x, y) \
                            for x, y in zip(get_map_x.flatten(), get_map_y.flatten())]),
                            get_map_x.shape)
        vtgt_y0 = np.reshape(np.array([self.vtgt_interp_y(x, y) \
                            for x, y in zip(get_map_x.flatten(), get_map_y.flatten())]),
                            get_map_x.shape)

        vtgt_x = np.cos(th)*vtgt_x0 - np.sin(th)*vtgt_y0
        vtgt_y = np.sin(th)*vtgt_x0 + np.cos(th)*vtgt_y0

        # debug
        import matplotlib.pyplot as plt
        plt.figure(100)
        plt.axes([.025, .025, .95, .95])
        plt.plot(get_map_x, get_map_y, '.')
        plt.axis('equal')

        plt.figure(101)
        plt.axes([.025, .025, .95, .95])
        R = np.sqrt(vtgt_x0**2 + vtgt_y0**2)
        plt.quiver(get_map_x, get_map_y, vtgt_x0, vtgt_y0, R)
        plt.axis('equal')

        plt.show()

        import pdb; pdb.set_trace()

        return np.stack((vtgt_x, vtgt_y))

# -----------------------------------------------------------------------------------------------------------------
    def get_vtgt(self, xy):
        vtgt_x = self.vtgt_interp_x(xy[0], xy[1])
        vtgt_y = self.vtgt_interp_y(xy[0], xy[1])
        return np.array([vtgt_x, vtgt_y])

# -----------------------------------------------------------------------------------------------------------------
    def _generate_grid(self, rng_xy=np.array([[-10, 10], [-10, 10]]), res=2):
        xo = .5*(rng_xy[0,0]+rng_xy[0,1])
        x_del = (rng_xy[0,1]-xo)*res
        yo = .5*(rng_xy[1,0]+rng_xy[1,1])
        y_del = (rng_xy[1,1]-yo)*res
        grid = np.mgrid[-x_del:x_del+1, -y_del:y_del+1]/res
        grid[0] = grid[0] + xo
        grid[1] = grid[1] + yo
        return grid

# -----------------------------------------------------------------------------------------------------------------
    def _set_sink_vtgt_amp(self, p_sink, d_sink, v_amp_rng, v_phase0, d_dec = 1):
        # d_dec: start to decelerate within d_dec of sink

        for i_x, x in enumerate(self.map[0,:,0]):
            for i_y, y in enumerate(self.map[1,0,:]):
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

                amp_norm = np.linalg.norm(self.vtgt[:,i_x,i_y])
                self.vtgt[0,i_x,i_y] = v_amp*self.vtgt[0,i_x,i_y]/amp_norm
                self.vtgt[1,i_x,i_y] = v_amp*self.vtgt[1,i_x,i_y]/amp_norm
        
