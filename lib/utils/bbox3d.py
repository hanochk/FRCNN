# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Sergey Karayev
# --------------------------------------------------------

import numpy as np
from utils.cython_bbox import bbox_overlaps
from utils.timer import Timer


def bbox_2d_overlap_multiple(boxes, angles, query_boxes, query_angles):
    N = boxes.shape[0]
    K = query_boxes.shape[0]

    overlaps = np.zeros((N, K))

    for box_idx in xrange(N):
        #        for query_idx in xrange(K):
        overlaps[box_idx, :] = bbox_2d_overlap(boxes[box_idx, :], angles[box_idx],
                                               query_boxes, query_angles)

    return overlaps


def bbox_2d_overlap(in_box, in_angle, in_query_boxes, in_query_angles):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float (x,y,wx,wy), with x,y being the center point
    angles: (N) ndarray of float, rotational angle of box in XY plane
    query_boxs: (K, 4) ndarray of float (x,y,wx,wy), with x,y being the center point
    query_angles: (N) ndarray of float, rotational angle of query_box in XY plane
    Returns
    -------
    overlaps: float, overlap of both boxes
    """
    box, query_boxes = np.atleast_2d(in_box, in_query_boxes)
    angle, query_angles = np.atleast_1d(in_angle, in_query_angles)

    overlaps = np.zeros((box.shape[0], query_boxes.shape[0]))

    # rotate both boxes such that box is aligned to both axes
    rotated_boxes = rotate_box(box, -angle)
    rotated_query_boxes = rotate_box(query_boxes, -angle)
    new_query_angles = query_angles - angle

    # solve cases with zero rotation using c overlaps code
    zero_angles_inds = np.where(new_query_angles == 0.)[0]
    overlaps[:, zero_angles_inds] = bbox_overlaps(ctrwh2tlbr(rotated_boxes).astype(np.double),
                                                  ctrwh2tlbr(rotated_query_boxes[zero_angles_inds, :]).astype(np.double))

    # solve cases with 90 degrees rotation using c overlaps code
    angle_pos90_inds = np.where(new_query_angles == np.pi / 2.)[0]
    angle_neg90_inds = np.where(new_query_angles == -np.pi / 2.)[0]
    if angle_neg90_inds.size > 0 or angle_pos90_inds.size > 0:
        angle_90_inds = np.concatenate(angle_pos90_inds, angle_neg90_inds)
        rotated_query_boxes[angle_90_inds, 2], rotated_query_boxes[angle_90_inds, 3] = \
            rotated_query_boxes[angle_90_inds, 3], rotated_query_boxes[angle_90_inds, 2]
        rotated_query_boxes[angle_pos90_inds, 1] += rotated_query_boxes[angle_pos90_inds, 3]
        rotated_query_boxes[angle_neg90_inds, 0] -= rotated_query_boxes[angle_neg90_inds, 2]
        overlaps[:, angle_90_inds] = \
            bbox_overlaps(ctrwh2tlbr(rotated_boxes).astype(np.double),
                          ctrwh2tlbr(rotated_query_boxes[angle_90_inds, :]).astype(np.double))[0, 0]

    # find indices of boxes with angles that are not +-90 or 0
    other_inds = np.where(np.logical_and(new_query_angles != np.pi / 2,
                                         np.logical_and(new_query_angles != -np.pi / 2,
                                                        new_query_angles != 0.)))[0]
    normal_rotated_query_boxes = rotated_query_boxes[other_inds, :]
    normal_new_query_angles = new_query_angles[other_inds]
    if other_inds.size > 0:
        overlaps[:, other_inds] = bbox_2d_overlap_approx(rotated_boxes, normal_rotated_query_boxes,
                                                         normal_new_query_angles)

    return overlaps


def bbox_2d_overlap_approx(rotated_boxes, rotated_query_boxes, new_query_angles):

    n_points = 50

    # create points to check intersection area
    ctr_boxes = rotated_boxes[:, :2]
    width_boxes = rotated_boxes[:, 2:]
    x_points = np.arange(ctr_boxes[:, 0] - width_boxes[:, 0] * 0.5,
                         ctr_boxes[:, 0] + width_boxes[:, 0] * 0.5,
                         width_boxes[:, 0] / float(n_points)) + 1e-5
    x_points = x_points[:n_points+1]
    y_points = np.arange(ctr_boxes[:, 1] - width_boxes[:, 1] * 0.5,
                         ctr_boxes[:, 1] + width_boxes[:, 1] * 0.5,
                         width_boxes[:, 1] / float(n_points)) + 1e-5
    y_points = y_points[:n_points+1]
    x_points, y_points = np.meshgrid(x_points, y_points, indexing='ij')

    horizontal_edge_direction = np.stack((np.cos(new_query_angles), np.sin(new_query_angles)), axis=1)  # rightwards
    vertical_edge_direction = np.stack((-np.sin(new_query_angles), np.cos(new_query_angles)), axis=1)  # upwards

    ctr_query_boxes = rotated_query_boxes[:, :2]
    width_query_boxes = rotated_query_boxes[:, 2:]
    bottom_left_query_box = ctr_query_boxes[:, :2] \
                            - width_query_boxes[:, 0][:, np.newaxis] * horizontal_edge_direction * 0.5 \
                            - width_query_boxes[:, 1][:, np.newaxis] * vertical_edge_direction

    bottom_right_query_box = bottom_left_query_box[:, :2] \
                             + width_query_boxes[:, 0][:, np.newaxis] * horizontal_edge_direction
    top_left_query_box = bottom_left_query_box[:, :2] \
                             + width_query_boxes[:, 1][:, np.newaxis] * vertical_edge_direction
    top_right_query_box = bottom_right_query_box + width_query_boxes[:, 1][:, np.newaxis] * vertical_edge_direction
    corners = np.stack((top_right_query_box, bottom_left_query_box,
                        bottom_right_query_box, top_left_query_box))
    max_vals = corners.max(axis=0)
    min_vals = corners.min(axis=0)

    horizontal_slope = horizontal_edge_direction[:, 1] / horizontal_edge_direction[:, 0]
    vertical_slope = vertical_edge_direction[:, 1] / vertical_edge_direction[:, 0]

    anchor1 = top_left_query_box
    anchor2 = bottom_right_query_box
    neg_slope_idcs = horizontal_slope < 0.
    anchor1[neg_slope_idcs] = top_right_query_box[horizontal_slope < 0.]
    anchor2[neg_slope_idcs] = bottom_left_query_box[horizontal_slope < 0.]

    for query_idx in xrange(anchor1.shape[0]):
        x_points[x_points == anchor1[query_idx, 0]] += 1e-5
        x_points[x_points == anchor2[query_idx, 0]] -= 1e-5

    slope1 = (y_points[..., np.newaxis] - anchor1[:, 1][np.newaxis, np.newaxis, :]) / \
             (x_points[..., np.newaxis] - anchor1[:, 0][np.newaxis, np.newaxis, :])
    slope2 = (y_points[..., np.newaxis] - anchor2[:, 1][np.newaxis, np.newaxis, :]) / \
             (x_points[..., np.newaxis] - anchor2[:, 0][np.newaxis, np.newaxis, :])

    slope1[:, :, neg_slope_idcs] = -slope1[:, :, neg_slope_idcs]
    slope2[:, :, neg_slope_idcs] = -slope2[:, :, neg_slope_idcs]

    within_rectangle = (slope1 <= horizontal_slope) & (slope1 >= vertical_slope) & \
                       (slope2 <= horizontal_slope) & (slope2 >= vertical_slope)

    within_rectangle[(x_points[..., np.newaxis] < min_vals[np.newaxis, np.newaxis, :, 0]) |
                     (y_points[..., np.newaxis] < min_vals[np.newaxis, np.newaxis, :, 1]) |
                     (x_points[..., np.newaxis] > max_vals[np.newaxis, np.newaxis, :, 0]) |
                     (y_points[..., np.newaxis] > max_vals[np.newaxis, np.newaxis, :, 1])] = False

    intersect = (within_rectangle.reshape((n_points ** 2, -1)).sum(axis=0).astype(np.float)) / \
                (n_points ** 2) * width_boxes[:, 0] * width_boxes[:, 1]
    area_boxes = width_boxes.prod(axis=1)
    area_query_boxes = width_query_boxes.prod(axis=1)
    union = area_boxes + area_query_boxes - intersect
    overlaps = intersect / union

    # import matplotlib.pyplot as plt
    # plt.scatter(x_points.ravel(), y_points.ravel(), c=within_rectangle.ravel())
    # plt.show()
    return overlaps


def rotate_box(boxes, angle):
    boxes[:, :2] = rotate_points(boxes[:, :2].view(), angle)
    return boxes


def rotate_points(pts, angle, ctr=np.array([0., 0.])):
    '''
    Rotate a point around a certain axis
    :param pts: points to rotate of dimension Nx2
    :param angle: angle of rotation in radians
    :param ctr: axis of rotation
    :return: rotated point
    '''
    pts_offset = pts - ctr[np.newaxis, :]
    rot_mat = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]).squeeze()
    pts = np.dot(pts_offset, rot_mat) + ctr
    return pts


# convert box coordinates from top left width height to top left bottom right
def tlwh2tlbr(box):
    out_box = in_box = np.atleast_2d(box.copy())
    out_box[:, 2] = in_box[:, 0] + in_box[:, 2] - 1.
    out_box[:, 3] = in_box[:, 1] + in_box[:, 3] - 1.
    return out_box


def ctrwh2tlbr(box):
    out_box = np.zeros_like(box, np.float32)
    out_box[:, 0] = box[:, 0] - 0.5 * box[:, 2]
    out_box[:, 1] = box[:, 1] - 0.5 * box[:, 3]
    out_box[:, 2] = out_box[:, 0] + box[:, 2] - 1.0
    out_box[:, 3] = out_box[:, 1] + box[:, 3] - 1.0
    return out_box


if __name__ == '__main__':
    box = np.array([0., 0., 10., 10.])
    query_box = np.array([1., 1., 1., 1.])
    boxes = np.tile(box, (11, 1))
    angles = np.zeros(11)  # np.array([0. * np.pi / 180.])
    boxes2 = np.tile(query_box, (10, 1))
    angles2 = np.arange(10) * 1. * np.pi / 180.  # np.array([0. * np.pi / 180.])
    print bbox_2d_overlap_multiple(boxes, angles, boxes2, angles2)
