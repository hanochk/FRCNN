import numpy as np
cimport numpy as np

def pc_maps(np.ndarray[np.float64_t, ndim=2] point_cloud, np.ndarray[np.float64_t, ndim=2] ranges,
            np.ndarray[np.float64_t, ndim=1] resolutions):

        # find the number of elements in each axis
        cdef np.ndarray[np.int64_t, ndim=1] numels = ((ranges[1, :] - ranges[0, :]) / resolutions).astype(np.int64)

        # initialize views
        cdef np.ndarray[np.int64_t, ndim=2] count_view = np.ones(numels[:2], dtype=np.int64)
        cdef np.ndarray[np.float64_t, ndim=3] height_view = -100. * np.ones(numels, dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=2] max_height_view = -100. * np.ones(numels[:2], dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=2] ref_view = np.zeros(numels[:2], dtype=np.float64)

        # A map signifying this voxel has no points. Initialized with True.
        cdef np.ndarray[np.int64_t, ndim=3] height_no_pts = np.ones_like(height_view, dtype=np.int64)

        # get indices of each point in the occupancy grid
        cdef np.ndarray[np.int64_t, ndim=2] idxs = ((point_cloud[:, :3] - ranges[0, :]) / resolutions).astype(np.int64)
        # points that are within the desired range
        cdef np.ndarray[np.int64_t, ndim=1] valid_point_cloud_idxs = np.where(((idxs >= 0) & (idxs < numels)).all(axis=1))[0]
        cdef np.ndarray[np.int64_t, ndim=2] original_idxs = idxs.copy()
        cdef np.ndarray[np.int64_t, ndim=1] idx

        cdef np.ndarray[np.float64_t, ndim=2] density_view
        cdef np.ndarray[np.int64_t, ndim=1] height_no_pts_num
        cdef int total_pts
        cdef np.ndarray[np.int64_t, ndim=2] no_pts
        cdef int no_pts_num
        cdef np.ndarray[np.float64_t, ndim=3] output

        point_cloud = point_cloud[valid_point_cloud_idxs]
        idxs = idxs[valid_point_cloud_idxs, :]

        # fill occupancy grids
        for i in range(idxs.shape[0]):
            idx = idxs[i]
            height_no_pts[idx[0], idx[1], idx[2]] = 0
            count_view[idx[0], idx[1]] += 1
            if max_height_view[idx[0], idx[1]] < point_cloud[i, 2]:
                max_height_view[idx[0], idx[1]] = point_cloud[i, 2]
                ref_view[idx[0], idx[1]] = point_cloud[i, 3]
            if height_view[idx[0], idx[1], idx[2]] < point_cloud[i, 2]:
                height_view[idx[0], idx[1], idx[2]] = point_cloud[i, 2]

        # change from count view to density view
#        density_view = np.log(count_view) / np.log(64)
#        density_view[density_view > 1.] = 1.

        # reduce means
#        height_view[height_no_pts == 1] = 0.
#        height_no_pts_num = height_no_pts.sum(axis=(0, 1))
#        total_pts = height_view.size
#        height_view -= np.mean(height_view, axis=(0, 1), keepdims=True) * \
#                       total_pts / (total_pts - height_no_pts_num[np.newaxis, np.newaxis, :])
#        height_view[height_no_pts == 1] = 0.
#        no_pts = np.zeros((height_no_pts.shape[0], height_no_pts.shape[1]), dtype=np.int)
#        no_pts[height_no_pts.sum(axis=2) == height_no_pts.shape[2]] = 1
#        ref_view[no_pts] = 0.
#        no_pts_num = no_pts.sum()
#        ref_view -= np.mean(ref_view) * total_pts / (total_pts - no_pts_num)
#        ref_view[no_pts] = 0.
#        density_view -= np.mean(density_view)

#        output = np.concatenate([density_view[:, :, np.newaxis], height_view, ref_view[:, :, np.newaxis]], axis=2)
        return height_view, count_view, ref_view, height_no_pts, idxs

def pc_maps_short(np.ndarray[np.float64_t, ndim=2] point_cloud, np.ndarray[np.float64_t, ndim=2] ranges,
            np.ndarray[np.float64_t, ndim=1] resolutions):

        # find the number of elements in each axis
        cdef np.ndarray[np.int64_t, ndim=1] numels = ((ranges[1, :] - ranges[0, :]) / resolutions).astype(np.int64)

        # initialize views
        cdef np.ndarray[np.int64_t, ndim=2] count_view = np.ones(numels[:2], dtype=np.int64)
        cdef np.ndarray[np.float64_t, ndim=3] height_view = -100. * np.ones(numels, dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=2] max_height_view = -100. * np.ones(numels[:2], dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=2] ref_view = np.zeros(numels[:2], dtype=np.float64)

        # A map signifying this voxel has no points. Initialized with True.
        cdef np.ndarray[np.int64_t, ndim=3] height_no_pts = np.ones_like(height_view, dtype=np.int64)

        # get indices of each point in the occupancy grid
        cdef np.ndarray[np.int64_t, ndim=2] idxs = ((point_cloud[:, :3] - ranges[0, :]) / resolutions).astype(np.int64)
        # points that are within the desired range
        cdef np.ndarray[np.int64_t, ndim=1] valid_point_cloud_idxs = np.where(((idxs >= 0) & (idxs < numels)).all(axis=1))[0]

        cdef np.ndarray[np.int64_t, ndim=1] idx

        cdef np.ndarray[np.float64_t, ndim=2] density_view
        cdef np.ndarray[np.int64_t, ndim=1] height_no_pts_num
        cdef int total_pts
        cdef np.ndarray[np.int64_t, ndim=2] no_pts
        cdef int no_pts_num
        cdef np.ndarray[np.float64_t, ndim=3] output

        point_cloud = point_cloud[valid_point_cloud_idxs]
        idxs = idxs[valid_point_cloud_idxs, :]

        # fill occupancy grids
        for i in range(idxs.shape[0]):
            idx = idxs[i]
            height_no_pts[idx[0], idx[1], idx[2]] = 0
            count_view[idx[0], idx[1]] += 1
            if max_height_view[idx[0], idx[1]] < point_cloud[i, 2]:
                max_height_view[idx[0], idx[1]] = point_cloud[i, 2]
                ref_view[idx[0], idx[1]] = point_cloud[i, 3]
            if height_view[idx[0], idx[1], idx[2]] < point_cloud[i, 2]:
                height_view[idx[0], idx[1], idx[2]] = point_cloud[i, 2]

        return height_view, count_view, ref_view, height_no_pts, idxs