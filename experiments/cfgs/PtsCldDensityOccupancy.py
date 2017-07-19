import matplotlib.pyplot as plt
import numpy as np
occ_res = 0.1
min_dist = 0.5
max_dist = 100
res_angle = 0.1
depth = np.arange(min_dist,max_dist,occ_res, dtype=float)
pts_per_gridcell = occ_res/(depth*res_angle/180*np.pi)
plt.figure(1)
plt.plot(depth,pts_per_gridcell)
plt.grid()
plt.title('Point cloud density per occupancy grid cell')
