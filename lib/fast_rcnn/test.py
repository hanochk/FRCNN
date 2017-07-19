from fast_rcnn.config import cfg, get_output_dir
import argparse
from utils.timer import Timer
import numpy as np
import cv2
from utils.cython_nms import nms, nms_new
from utils.boxes_grid import get_boxes_grid
import cPickle
import heapq
from utils.blob import im_list_to_blob
import os
import math
from rpn_msr.generate import imdb_proposals_det
import tensorflow as tf
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv, bbox3d_transform_inv
import matplotlib.pyplot as plt
from tensorflow.python.client import timeline
import time
from utils.math_utils import get_3d_box_top_view


def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
        im (ndarray): a color image in BGR order
    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


def _get_pc_blob(im):
    """Builds an input blob from the point clouds in the roidb at the specified
    scales.
    """
    processed_ims = [im]
    im_scales = [1.]

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales


def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.
    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob
    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)


def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.
    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob
    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)
    scales = np.array(scales)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels


def _get_blobs(im, rois, modality):
    """Convert an image and RoIs within that image into network inputs."""
    if cfg.TEST.HAS_RPN:
        blobs = {'data': None, 'rois': None}
        if modality == 'image':
            blobs['data'], im_scale_factors = _get_image_blob(im)
        elif modality == 'point_cloud':
            blobs['data'], im_scale_factors = _get_pc_blob(im)
    else:
        blobs = {'data': None, 'rois': None}
        blobs['data'], im_scale_factors = _get_image_blob(im)
        if cfg.IS_MULTISCALE:
            if cfg.IS_EXTRAPOLATING:
                blobs['rois'] = _get_rois_blob(rois, cfg.TEST.SCALES)
            else:
                blobs['rois'] = _get_rois_blob(rois, cfg.TEST.SCALES_BASE)
        else:
            blobs['rois'] = _get_rois_blob(rois, cfg.TEST.SCALES_BASE)

    return blobs, im_scale_factors


def _clip_boxes(boxes, im_shape):
    """Clip boxes to image boundaries."""
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
    return boxes


def _clip_boxes3d(boxes, im_shape):
    """Clip boxes to image boundaries."""
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
    return boxes


def _rescale_boxes(boxes, inds, scales):
    """Rescale boxes according to image rescaling."""

    for i in xrange(boxes.shape[0]):
        boxes[i, :] = boxes[i, :] / scales[int(inds[i])]

    return boxes


def im_detect(sess, net, im, boxes=None, modality='point_cloud'):
    """Detect object classes in an image given object proposals.
    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals
    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
        modality: 'point_cloud' or 'image'
    """

    blobs, im_scales = _get_blobs(im, boxes, modality)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(hashes, return_index=True,
                                        return_inverse=True)
        blobs['rois'] = blobs['rois'][index, :]
        boxes = boxes[index, :]

    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
            dtype=np.float32)
    # forward pass
    if cfg.TEST.HAS_RPN:
        feed_dict = {net.data: blobs['data'], net.im_info: blobs['im_info'], net.keep_prob: 1.0}
    else:
        feed_dict = {net.data: blobs['data'], net.rois: blobs['rois'], net.keep_prob: 1.0}

    run_options = None
    run_metadata = None
    if cfg.TEST.DEBUG_TIMELINE:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

    cls_score, cls_prob, bbox_pred, rois = sess.run(
        [net.get_output('cls_score'), net.get_output('cls_prob'), net.get_output('bbox_pred'), net.get_output('rois')],
        feed_dict=feed_dict,
        options=run_options,
        run_metadata=run_metadata)

    if cfg.TEST.HAS_RPN:
        assert len(im_scales) == 1, "Only single-image batch implemented"
        boxes = rois[:, 1:5] / im_scales[0]

    if cfg.TEST.SVM:
        # use the raw scores before softmax under the assumption they
        # were trained as linear SVMs
        scores = cls_score
    else:
        # use softmax estimated probabilities
        scores = cls_prob

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred
        if modality == 'image':
            pred_boxes = bbox_transform_inv(boxes, box_deltas)
            pred_boxes = _clip_boxes(pred_boxes, im.shape)
        elif modality == 'point_cloud':
            pred_boxes_3d = bbox3d_transform_inv(boxes, box_deltas)
            classes = scores.argmax(axis=1)
            row_inds = np.arange(pred_boxes_3d.shape[0])[:, np.newaxis]
            col_inds = 7 * classes[:, np.newaxis] + np.arange(7)[np.newaxis, :]
            pred_boxes_3d = pred_boxes_3d[row_inds, col_inds]
            pred_boxes, keep_inds, _ = get_3d_box_top_view(pred_boxes_3d)
            scores = scores[keep_inds]
            #pred_boxes = _clip_boxes(pred_boxes_2d, im.shape)

    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.DEDUP_BOXES > 0 and not cfg.TEST.HAS_RPN:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]

    if cfg.TEST.DEBUG_TIMELINE:
        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        trace_file = open(str(long(time.time() * 1000)) + '-test-timeline.ctf.json', 'w')
        trace_file.write(trace.generate_chrome_trace_format(show_memory=False))
        trace_file.close()

    return scores, pred_boxes


def vis_detections(im, class_name, dets, color='w', thresh=0.8):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    # im = im[:, :, (2, 1, 0)]
    for i in xrange(np.minimum(1000, dets.shape[0])):
        bbox = dets[i, :4]
        score = dets[i, -1]
        if score > thresh:
            # plt.cla()
            # plt.imshow(im)
            plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor=color, linewidth=1)
            )
            # plt.gca().text(bbox[0], bbox[1] - 2,
            #                '{:s} {:.3f}'.format(class_name, score),
            #                bbox=dict(facecolor='blue', alpha=0.5),
            #                fontsize=14, color='white')

            plt.title('{}  {:.3f}'.format(class_name, score))
            # plt.show()


def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(num_classes)]
    for cls_ind in xrange(num_classes):
        for im_ind in xrange(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue

            x1 = dets[:, 0]
            y1 = dets[:, 1]
            x2 = dets[:, 2]
            y2 = dets[:, 3]
            scores = dets[:, 4]
            inds = np.where((x2 > x1) & (y2 > y1) & (scores > cfg.TEST.DET_THRESHOLD))[0]
            dets = dets[inds, :]
            if dets == []:
                continue

            keep = nms(dets, thresh)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes


def test_net(sess, net, imdb, weights_filename, max_per_image=300, thresh=0.05, vis=False):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, weights_filename)
    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}

    for i in xrange(num_images):
        _t['misc'].tic()
        roidb = imdb.roidb
        # filter out any ground truth boxes
        box_proposals = None

        if imdb.modality == 'image':
            im = cv2.imread(imdb.image_path_at(i))
        elif imdb.modality == 'point_cloud':
            with open(imdb.image_path_at(i), 'rb') as fid:
                im = cPickle.load(fid)
        _t['misc'].toc()
        _t['im_detect'].tic()
        scores, boxes = im_detect(sess, net, im, box_proposals, imdb.modality)
        _t['im_detect'].toc()

        _t['misc'].tic()
        if vis:
            if imdb.modality == 'image':
                image = im[:, :, (2, 1, 0)]
            elif imdb.modality == 'point_cloud':
                image = im[:, :, 0]
            plt.cla()
            plt.imshow(image)

        # skip j = 0, because it's the background class
        classes = scores.argmax(axis=1)
        winning_scores = scores.max(axis=1)
        inds = np.where((winning_scores > thresh) & (classes != 0))[0]
        cls_scores = winning_scores[inds]
        cls_boxes = boxes[inds]
        cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
            .astype(np.float32, copy=False)
        if cls_dets.size > 0:
            keep = nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep, :]
        if vis:
            vis_detections(image, '', cls_dets)
        gt_boxes = get_3d_box_top_view(roidb[i]['boxes_3d'])[0]
        gt_boxes = np.hstack((gt_boxes, np.ones((gt_boxes.shape[0], 1))))
        vis_detections(image, '', gt_boxes, color='r')
        if vis:
            plt.show()
        _t['misc'].toc()

        time_left = (_t['im_detect'].average_time + _t['misc'].average_time) * (num_images - i)
        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s time left: {}' \
            .format(i + 1, num_images, _t['im_detect'].average_time,
                    _t['misc'].average_time, time.strftime("%Hh %Mm %Ss", time.gmtime(time_left)))

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    print 'Evaluating detections'
    imdb.evaluate_detections(all_boxes, output_dir)
