# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network."""

from fast_rcnn.config import cfg
import gt_data_layer.roidb as gdl_roidb
import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.layer import RoIDataLayer
from utils.timer import Timer
import numpy as np
import os
import tensorflow as tf
import sys
from tensorflow.python.client import timeline
import time
import datetime
from utils.moving_averages import ExponentialMovingAverage
from fast_rcnn.bbox_transform import clip_boxes, bbox_transform_inv, bbox3d_transform_inv
from utils.cython_bbox import bbox_overlaps
from utils.math_utils import get_3d_box_top_view
import matplotlib.pyplot as plt
import utils.display_tools as dsptls


class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, sess, saver, network, imdb, roidb, imdb_val, roidb_val, output_dir, pretrained_model=None,
                 modality='image'):
        """Initialize the SolverWrapper."""
        self.net = network
        self.imdb = imdb
        self.roidb = roidb
        self.imdb_val = imdb_val
        self.roidb_val = roidb_val
        self.output_dir = output_dir
        self.pretrained_model = pretrained_model
        self._modality = modality

        print 'Computing bounding-box regression targets...'
        if cfg.TRAIN.BBOX_REG:
            self.bbox_means, self.bbox_stds = rdl_roidb.add_bbox_regression_targets(roidb)
        print 'done'

        log_folder = os.path.join(cfg.TRAIN.LOG_DIR,'run_{}'.format(datetime.datetime.now().strftime("%y%m%d_%H%M%S")))
        self.summary_writer = tf.summary.FileWriter(log_folder, sess.graph)
        print 'Writing log to {}'.format(log_folder)

        # For checkpoint
        self.saver = saver

    def snapshot(self, sess, iter):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.net

        if cfg.TRAIN.BBOX_REG and net.layers.has_key('bbox_pred'):
            # save original values
            with tf.variable_scope('bbox_pred/fc', reuse=True):
                weights = tf.get_variable("weights")
                biases = tf.get_variable("biases")

            orig_0 = weights.eval()
            orig_1 = biases.eval()

            # scale and shift with bbox reg unnormalization; then save snapshot
            weights_shape = weights.get_shape().as_list()
            sess.run(net.bbox_weights_assign,
                     feed_dict={net.bbox_weights: orig_0 * np.tile(self.bbox_stds, (weights_shape[0], 1))})
            sess.run(net.bbox_bias_assign, feed_dict={net.bbox_biases: orig_1 * self.bbox_stds + self.bbox_means})

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (cfg.TRAIN.SNAPSHOT_PREFIX + infix +
                    '_iter_{:d}'.format(iter + 1) + '.ckpt')
        filename = os.path.join(self.output_dir, filename)

        self.saver.save(sess, filename)
        print 'Wrote snapshot to: {:s}'.format(filename)

        if cfg.TRAIN.BBOX_REG and net.layers.has_key('bbox_pred'):
            with tf.variable_scope('bbox_pred', reuse=True):
                # restore net to original state
                sess.run(net.bbox_weights_assign, feed_dict={net.bbox_weights: orig_0})
                sess.run(net.bbox_bias_assign, feed_dict={net.bbox_biases: orig_1})

    def train_model(self, sess, max_iters):
        """Network training loop."""

        data_layer = get_data_layer(self.roidb, self.imdb.num_classes, self._modality)
        data_layer_val = get_data_layer(self.roidb_val, self.imdb.num_classes, self._modality)

        self.net.setup_training(self.bbox_means, self.bbox_stds)

        # initialize variables
        sess.run(tf.global_variables_initializer())
        if self.pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(self.pretrained_model)
            self.net.load(self.pretrained_model, sess, self.saver, True)

        last_snapshot_iter = -1
        timer = Timer()
        last_update_lr_step = self.net.train_dict['last_update_lr_step']
        lr = self.net.train_dict['lr']
        curr_lr = sess.run(lr)
        for iter in range(max_iters):
            # get one batch
            blobs = data_layer.forward()

            # update learning rate
            if iter - last_update_lr_step > cfg.TRAIN.STEPSIZE:
                curr_lr = sess.run(lr)
                sess.run(tf.assign(lr, curr_lr * cfg.TRAIN.LEARNING_RATE_DECAY))
                self.net.train_dict['last_update_lr_step'] = last_update_lr_step = iter

            timer.tic()
            if iter % cfg.TRAIN.DISPLAY == 0:
                loss_val, loss_val_smoothed, summary, run_metadata = self.net.train_step_summary(sess, blobs)
                self.summary_writer.add_summary(summary, iter)
                if run_metadata is not None:
                    self.summary_writer.add_run_metadata(run_metadata, 'iter{}'.format(iter), iter)
            else:
                loss_val, loss_val_smoothed, summary = self.net.train_step(sess, blobs)
                if iter % cfg.TRAIN.RUN_TESTS == 0:
                    self.summary_writer.add_summary(summary, iter)

            if iter % cfg.TRAIN.RUN_TESTS == 0:
                precision, recall, error_rate, loss_val = self.net.test(sess, blobs)
                results_summary = tf.Summary(value=[tf.Summary.Value(tag='train_recall', simple_value=recall),
                                                    tf.Summary.Value(tag='train_precision', simple_value=precision),
                                                    tf.Summary.Value(tag='train_error_rate', simple_value=error_rate),
                                                    tf.Summary.Value(tag='train_loss', simple_value=loss_val)])
                self.summary_writer.add_summary(results_summary, iter)
                blobs_val = data_layer_val.forward()
                validation_precision, validation_recall, validation_error_rate, validation_loss = self.net.test(sess, blobs_val)
                results_summary = \
                    tf.Summary(value=[tf.Summary.Value(tag='validation_recall', simple_value=validation_recall),
                                      tf.Summary.Value(tag='validation_precision', simple_value=validation_precision),
                                      tf.Summary.Value(tag='validation_error_rate', simple_value=validation_error_rate),
                                      tf.Summary.Value(tag='validation_loss', simple_value=validation_loss)
                                      ])
                self.summary_writer.add_summary(results_summary, iter)

            if iter % cfg.TRAIN.RUN_TESTS == 0:
                self.summary_writer.flush()

            timer.toc()

            if iter % cfg.TRAIN.DISPLAY == 0:
                time_left = (max_iters - iter) * timer.average_time
                print 'iter: %d / %d, total loss: %.4f, total_loss filtered: %.4f, recall: %.3f, precision %.3f, lr: %f, error_rate %f' % \
                      (iter + 1, max_iters, loss_val, loss_val_smoothed, recall, precision, curr_lr, error_rate)
                print 'speed: {:.3f}s / iter, time left: {}'.format(timer.average_time, time.strftime("%jd %Hh %Mm %Ss",
                                                                                                      time.gmtime(
                                                                                                          time_left)))
                if iter == 10 * cfg.TRAIN.DISPLAY:
                    timer.reset()

            if iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                self.snapshot(sess, iter)

        if last_snapshot_iter != iter:
            self.snapshot(sess, iter)
        return loss_val_smoothed

    def create_summaries(self, loss, cross_entropy_loss, loss_box, rpn_ce_loss, rpn_loss_box):
        rpn_cls = self.net.get_output('rpn_cls_score')
        input_data = self.net.get_output('data')
        with tf.name_scope('summaries'):
            with tf.name_scope('rpn'):
                # for i in range(0, 30):
                #     tf.summary.image('rpn_cls_score_map_{}'.format(i), tf.slice(rpn_cls, [0, 0, 0, i], [1, -1, -1, 1]))
                for i in range(15):
                    for j in range(2):
                        tf.summary.image('rpn_cls_prob_{}_{}'.format(i, j == 1),
                                         tf.slice(tf.reshape(self.net.get_output('rpn_cls_prob'), (15, 64, 32, 2)),
                                                  [i, 0, 0, j], [1, -1, -1, 1]))
                tf.summary.image('input_image', tf.slice(input_data, [0, 0, 0, 0], [1, -1, -1, 1]))
                # tf.summary.histogram('rpn_cls_hist', rpn_cls)
                tf.summary.scalar('rpn_ce_loss', rpn_ce_loss)
                tf.summary.scalar('cross_entropy_loss', cross_entropy_loss)
                tf.summary.scalar('loss_box', loss_box)
                tf.summary.scalar('rpn_loss_box', rpn_loss_box)


def train_net(network, imdb, roidb, imdb_val, roidb_val, output_dir, pretrained_model=None, max_iters=40000,
              modality='point_cloud'):
    """Train a Fast R-CNN network."""
    roidb = filter_roidb(roidb)
    roidb_val = filter_roidb(roidb_val)
    saver = tf.train.Saver(max_to_keep=100)
    gpu_options = tf.GPUOptions(allow_growth=False)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)) as sess:
        sw = SolverWrapper(sess, saver, network, imdb, roidb, imdb_val, roidb_val, output_dir,
                           pretrained_model=pretrained_model, modality=modality)
        print 'Solving...'
        loss = sw.train_model(sess, max_iters)
        print 'done solving'
    return loss


def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
        print 'done'

    print 'Preparing training data...'
    rdl_roidb.prepare_roidb(imdb)
    print 'done'

    return imdb.roidb


def get_data_layer(roidb, num_classes, modality='image'):
    """return a data layer."""
    return RoIDataLayer(roidb, num_classes, modality=modality)


def filter_roidb(roidb):
    """Remove roidb entries that have no usable RoIs."""

    def is_valid(entry):
        # Valid images have:
        #   (1) At least one foreground RoI OR
        #   (2) At least one background RoI
        overlaps = entry['max_overlaps']
        # find boxes with sufficient overlap
        fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # image is only valid if such boxes exist
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        return valid

    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    print 'Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                       num, num_after)
    return filtered_roidb
