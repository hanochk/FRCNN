# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""

import numpy as np
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import bbox_transform
from fast_rcnn.bbox_transform import bbox3d_transform
from utils.cython_bbox import bbox_overlaps
from rpn_msr.generate_anchors import generate_offset_anchors
import PIL
import cPickle

def prepare_roidb(imdb):
    """Enrich the imdb's roidb by adding some derived quantities that
    are useful for training. This function precomputes the maximum
    overlap, taken over ground-truth boxes, between each ROI and
    each ground-truth box. The class with maximum overlap is also
    recorded.
    """
    roidb = imdb.roidb
    if imdb.modality == 'point_cloud':
        with open(imdb.image_path_at(0), 'rb') as fid:
            shape = cPickle.load(fid).shape

    for i in xrange(len(imdb.image_index)):
        if imdb.modality != 'point_cloud':
            shape = PIL.Image.open(imdb.image_path_at(i)).size
        roidb[i]['file_path'] = imdb.image_path_at(i)
        roidb[i]['width'] = shape[0]
        roidb[i]['height'] = shape[1]
        # need gt_overlaps as a dense array for argmax
        gt_overlaps = roidb[i]['gt_overlaps'].toarray()
        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = gt_overlaps.argmax(axis=1)
        roidb[i]['max_classes'] = max_classes
        roidb[i]['max_overlaps'] = max_overlaps
        # sanity checks
        # max overlap of 0 => class should be zero (background)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # max overlap > 0 => class should not be zero (must be a fg class)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)

def add_bbox_regression_targets(roidb):
    """Add information needed to train bounding-box regressors."""
    assert len(roidb) > 0
    assert 'max_classes' in roidb[0], 'Did you call prepare_roidb first?'

    num_images = len(roidb)
    # Infer number of classes from the number of columns in gt_overlaps
    num_classes = roidb[0]['gt_overlaps'].shape[1]
    im_info = (roidb[0]['height'], roidb[0]['width'])
    # generate all anchors of the image
    rois = generate_offset_anchors(cfg.NET.DOWNSAMPLE, cfg.NET.RPN_BASE_SIZE, cfg.NET.RPN_ANCHOR_RATIOS, cfg.NET.RPN_ANCHOR_SCALES, im_info)

    for im_i in xrange(num_images):
        gt_boxes_3d = roidb[im_i]['boxes_3d']
        gt_boxes_2d = roidb[im_i]['boxes']
        gt_classes = roidb[im_i]['gt_classes']
        roidb[im_i]['bbox_targets'] = _compute_targets(rois, gt_boxes_2d, gt_boxes_3d, gt_classes)

    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Use fixed / precomputed "means" and "stds" instead of empirical values
        means = np.tile(
                np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (num_classes, 1))
        stds = np.tile(
                np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (num_classes, 1))
    else:
        # Compute values needed for means and stds
        # var(x) = E(x^2) - E(x)^2
        class_counts = np.zeros((num_classes, 1)) + cfg.EPS
        sums = np.zeros((num_classes, 7))
        squared_sums = np.zeros((num_classes, 7))
        for im_i in xrange(num_images):
            targets = roidb[im_i]['bbox_targets']
            if targets.size == 0:
                continue
            for cls in xrange(1, num_classes):
                cls_inds = np.where(targets[:, 0] == cls)[0]
                if cls_inds.size > 0:
                    class_counts[cls] += cls_inds.size
                    sums[cls, :] += targets[cls_inds, 1:].sum(axis=0)
                    squared_sums[cls, :] += \
                            (targets[cls_inds, 1:] ** 2).sum(axis=0)

        means = sums / class_counts
        stds = np.sqrt(squared_sums / class_counts - means ** 2)

    print 'bbox target means:'
    print means
    print means[1:, :].mean(axis=0) # ignore bg class
    print 'bbox target stdevs:'
    print stds
    print stds[1:, :].mean(axis=0) # ignore bg class

    # Normalize targets
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS:
        print "Normalizing targets"
        for im_i in xrange(num_images):
            targets = roidb[im_i]['bbox_targets']
            for cls in xrange(1, num_classes):
                #cls_inds = np.where(targets[:, 0] == cls)[0]
                #roidb[im_i]['bbox_targets'][cls_inds, 1:] -= means[cls, :]
                #roidb[im_i]['bbox_targets'][cls_inds, 1:] /= stds[cls, :]
                pass
    else:
        print "NOT normalizing targets"

    # These values will be needed for making predictions
    # (the predicts will need to be unnormalized and uncentered)
    return means.ravel(), stds.ravel()


def _compute_targets(rois, gt_boxes_2d, gt_boxes_3d, gt_classes):
    """Compute bounding-box regression targets for an image."""
    # Indices of ground-truth ROIs
    # Get IoU overlap between each ex ROI and gt ROI
    ex_gt_overlaps = bbox_overlaps(
        np.ascontiguousarray(rois, dtype=np.float),
        np.ascontiguousarray(gt_boxes_2d, dtype=np.float))

    # Find which gt box each ex ROI has max overlap with:
    # this will be the ex ROI's gt target
    gt_assignment = ex_gt_overlaps.argmax(axis=1)
    max_overlaps = ex_gt_overlaps.max(axis=1)
    ex_inds = np.where(max_overlaps > cfg.TRAIN.BBOX_THRESH)[0]
    gt_boxes = gt_boxes_3d[gt_assignment[ex_inds], :]
    labels = gt_classes[gt_assignment[ex_inds]]

    if gt_boxes.size == 0:
        return np.array([])
    targets = np.zeros((ex_inds.shape[0], 8), dtype=np.float32)
    targets[:, 0] = labels
    targets[:, 1:] = bbox3d_transform(rois[ex_inds], gt_boxes)
    return targets
