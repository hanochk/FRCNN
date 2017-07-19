# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np
from fast_rcnn.config import cfg


def bbox_transform(ex_boxes, gt_boxes):
    ex_widths = ex_boxes[:, 2] - ex_boxes[:, 0] + 1.0
    ex_heights = ex_boxes[:, 3] - ex_boxes[:, 1] + 1.0
    ex_ctr_x = ex_boxes[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_boxes[:, 1] + 0.5 * ex_heights

    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0] + 1.0
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1] + 1.0
    gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets


def bbox_transform_inv(boxes, deltas):
    np.seterr(all='warn')
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]

    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes


def bbox3d_transform_inv(boxes, deltas, resolutions=np.array(cfg.RESOLUTIONS[:2]), origin2d=np.array(cfg.RANGES[0][:2])):
    """ converts 2d boxes in image coordinates to 3d boxes using deltas"""
    np.seterr(all='warn')
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    # create box equivalent in 3d coordinate system
    widths = (boxes[:, 2] - boxes[:, 0] + 1.0) * resolutions[1]
    lengths = (boxes[:, 3] - boxes[:, 1] + 1.0) * resolutions[0]
    heights = cfg.TRAIN.BBOX_3D_BIAS_HEIGHT * np.ones_like(widths)
    ctr_x = (boxes[:, 1]) * resolutions[0] + 0.5 * lengths + origin2d[1]
    ctr_y = (boxes[:, 0]) * resolutions[1] + 0.5 * widths + origin2d[0]
    ctr_z = cfg.TRAIN.BBOX_3D_BIAS_CENTER_Z * np.ones_like(ctr_x)

    dx = deltas[:, 0::7]
    dy = deltas[:, 1::7]
    dz = deltas[:, 2::7]
    dl = deltas[:, 3::7]
    dw = deltas[:, 4::7]
    dh = deltas[:, 5::7]
    pred_cosrz = deltas[:, 6::7]

    pred_ctr_x = dx * lengths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * widths[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_ctr_z = dz * heights[:, np.newaxis] + ctr_z[:, np.newaxis]

    pred_l = np.exp(dl) * lengths[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_3d_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x
    pred_3d_boxes[:, 0::7] = pred_ctr_x
    # y
    pred_3d_boxes[:, 1::7] = pred_ctr_y
    # z
    pred_3d_boxes[:, 2::7] = pred_ctr_z
    # wx
    pred_3d_boxes[:, 3::7] = pred_l
    # wy
    pred_3d_boxes[:, 4::7] = pred_w
    # wz
    pred_3d_boxes[:, 5::7] = pred_h
    # cos(rz)
    pred_3d_boxes[:, 6::7] = pred_cosrz

    return pred_3d_boxes


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """

    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes


def bbox3d_transform(ex_2d_boxes, gt_3d_boxes, resolutions=np.array(cfg.RESOLUTIONS[:2]), origin2d=np.array(cfg.RANGES[0][:2])):
    # gt_3d_boxes = (x,y,z,wx,wy,wz,cos(rz))
    # ex_2d_boxes = (x1,y1,x2,y2)
    # origin2d is the origin of the 2d image in the 3d coordinates
    # TODO: propagate resolutions and origin2d in outer code
    ex_widths = (ex_2d_boxes[:, 2] - ex_2d_boxes[:, 0] + 1.0) * resolutions[1]
    ex_heights = (ex_2d_boxes[:, 3] - ex_2d_boxes[:, 1] + 1.0) * resolutions[0]
    ex_ctr_x = ex_2d_boxes[:, 1] * resolutions[0] + 0.5 * ex_heights + origin2d[1]
    ex_ctr_y = ex_2d_boxes[:, 0] * resolutions[1] + 0.5 * ex_widths + origin2d[0]

    gt_ctr_x = gt_3d_boxes[:, 0]
    gt_ctr_y = gt_3d_boxes[:, 1]
    gt_bottom_z = gt_3d_boxes[:, 2]
    gt_lengths = gt_3d_boxes[:, 3]
    gt_widths = gt_3d_boxes[:, 4]
    gt_heights = gt_3d_boxes[:, 5]
    gt_cosrz = gt_3d_boxes[:, 6]

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_heights
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_widths
    targets_dz = (gt_bottom_z - cfg.TRAIN.BBOX_3D_BIAS_CENTER_Z) / cfg.TRAIN.BBOX_3D_BIAS_HEIGHT
    targets_dl = np.log(gt_lengths / ex_heights)
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / cfg.TRAIN.BBOX_3D_BIAS_HEIGHT)
    targets_dcosrz = gt_cosrz

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dz, targets_dl, targets_dw, targets_dh, targets_dcosrz)).transpose()
    return targets


if __name__ == '__main__':
    from utils.math_utils import get_3d_box_top_view
    ex_roi = np.array([[6.17556763, 110.91677094, 33.54301071, 130.847229]])
    gt_roi = np.array([[24.12875938, -22.26267052, -0.41540495, 4.94000006, 2.06999993, 2.05999994, 0.99980003]])
    original_2d = get_3d_box_top_view(gt_roi[0], resolutions=np.array(cfg.RESOLUTIONS), ranges=np.array(cfg.RANGES))
    delta = bbox3d_transform(ex_roi, gt_roi)
    res_3d = bbox3d_transform_inv(ex_roi, delta)
    res_2d = get_3d_box_top_view(res_3d[0], resolutions=np.array(cfg.RESOLUTIONS), ranges=np.array(cfg.RANGES))
    print "2d relative error: {}".format(np.linalg.norm(res_2d[0] - original_2d[0]) / np.linalg.norm(original_2d[0]))
    print "3d relative error: {}".format(np.linalg.norm(res_3d - gt_roi) / np.linalg.norm(gt_roi))
    pass