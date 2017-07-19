import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
import roi_pooling_layer.roi_pooling_op as roi_pool_op
import roi_pooling_layer.roi_pooling_op_grad
from rpn_msr.proposal_layer_tf import proposal_layer as proposal_layer_py
from rpn_msr.anchor_target_layer_tf import anchor_target_layer as anchor_target_layer_py
from rpn_msr.proposal_target_layer_tf import proposal_target_layer as proposal_target_layer_py
from rpn_msr.proposal_target_layer_3d_tf import proposal_target_layer_3d as proposal_target_layer_3d_py
from utils.math_utils import get_3d_box_top_view, bbox_reduce
from fast_rcnn.bbox_transform import bbox3d_transform_inv
from utils.cython_bbox import bbox_overlaps
from fast_rcnn.config import cfg
import utils.display_tools as dsptls
import matplotlib.pyplot as plt

DEFAULT_PADDING = 'SAME'

arg_scope = tf.contrib.framework.arg_scope


def layer(op):
    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.inputs) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.inputs) == 1:
            layer_input = self.inputs[0]
        else:
            layer_input = list(self.inputs)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated


class Network(object):
    def __init__(self, inputs, trainable=True):
        self.inputs = []
        self.layers = dict(inputs)
        self.trainable = trainable
        self._losses = {}
        self._predictions = {}
        self._proposal_targets = {}
        self._anchor_targets = {}
        self._train_summaries = []
        self._value_summaries = {}
        self._score_summaries = {}
        self._act_summaries = []
        self._ema_ops = {}
        self.train_dict = {}

    def setup(self):
        raise NotImplementedError('Must be subclassed.')

    def load(self, data_path, session, saver, ignore_missing=False):
        if data_path.endswith('.ckpt'):
            saver.restore(session, data_path)
        else:
            data_dict = np.load(data_path).item()
            for key in data_dict:
                with tf.variable_scope(key, reuse=True):
                    for subkey in data_dict[key]:
                        try:
                            var = tf.get_variable(subkey)
                            session.run(var.assign(data_dict[key][subkey]))
                            print "assign pretrain model " + subkey + " to " + key
                        except ValueError:
                            print "ignore " + key
                            if not ignore_missing:
                                raise

    def feed(self, *args):
        assert len(args) != 0
        self.inputs = []
        for layer in args:
            if isinstance(layer, basestring):
                try:
                    layer = self.layers[layer]
                    print layer
                except KeyError:
                    print self.layers.keys()
                    raise KeyError('Unknown layer name fed: %s' % layer)
            self.inputs.append(layer)
        return self

    def get_output(self, layer):
        try:
            layer = self.layers[layer]
        except KeyError:
            print self.layers.keys()
            raise KeyError('Unknown layer name fed: %s' % layer)
        return layer

    def get_unique_name(self, prefix):
        id = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, id)

    def make_var(self, name, shape, initializer=None, trainable=True, regularizer=None):
        return tf.get_variable(name, shape, initializer=initializer, trainable=trainable, regularizer=regularizer)

    def validate_padding(self, padding):
        assert padding in ('SAME', 'VALID')

    @layer
    def deconv(self, input, k_h, k_w, c_o, s_h, s_w, name, relu=True, trainable=True, padding=DEFAULT_PADDING,
               batch_norm=True, phase=False, bn_scale=True):
        if relu:
            activation_fn = tf.nn.relu
        else:
            activation_fn = None
        weight_regularizer = layers.l2_regularizer(cfg.TRAIN.REGULARIZER)
        if batch_norm:
            normalizer_fn = layers.batch_norm
            normalizer_params = {'is_training': phase, 'fused': True, 'scale': bn_scale}
        else:
            normalizer_fn = None
            normalizer_params = None

        out = layers.conv2d_transpose(inputs=input, num_outputs=c_o, kernel_size=(k_h, k_w), stride=(s_h, s_w),
                                      padding=padding, activation_fn=activation_fn,
                                      weights_regularizer=weight_regularizer, trainable=trainable, scope=name,
                                      normalizer_fn=normalizer_fn, normalizer_params=normalizer_params)
        return out

    @layer
    def conv(self, input, k_h, k_w, c_o, s_h, s_w, name, relu=True, trainable=True, padding=DEFAULT_PADDING,
               batch_norm=True, phase=False, bn_scale=True):
        if relu:
            activation_fn = tf.nn.relu
        else:
            activation_fn = None
        weight_regularizer = layers.l2_regularizer(cfg.TRAIN.REGULARIZER)
        if batch_norm:
            normalizer_fn = layers.batch_norm
            normalizer_params = {'is_training': phase, 'fused': True, 'scale': bn_scale}
        else:
            normalizer_fn = None
            normalizer_params = None
        out = layers.conv2d(inputs=input, num_outputs=c_o, kernel_size=(k_h, k_w), stride=(s_h, s_w),
                            padding=padding, activation_fn=activation_fn,
                            weights_regularizer=weight_regularizer, trainable=trainable, scope=name,
                            normalizer_fn=normalizer_fn, normalizer_params=normalizer_params)
        return out

    @layer
    def conv_old(self, input, k_h, k_w, c_o, s_h, s_w, name, relu=True, padding=DEFAULT_PADDING, group=1, trainable=True,
             batch_norm=True, phase=False):
        self.validate_padding(padding)
        c_i = input.get_shape()[-1]
        assert c_i % group == 0
        assert c_o % group == 0
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:
            init_weights = tf.contrib.layers.xavier_initializer()
            init_biases = tf.constant_initializer(0.0)
            kernel = self.make_var('weights', [k_h, k_w, c_i / group, c_o], init_weights, trainable,
                                   regularizer=layers.l2_regularizer(cfg.TRAIN.REGULARIZER))
            if group == 1:
                conv = convolve(input, kernel)
            else:
                input_groups = tf.split(3, group, input)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                conv = tf.concat(3, output_groups)

            if batch_norm:
                out = tf.contrib.layers.batch_norm(conv, center=True, scale=True, is_training=phase, scope='bn')
            else:
                biases = self.make_var('biases', [c_o], init_biases, trainable)
                out = tf.nn.bias_add(conv, biases)
            if relu:
                out = tf.nn.relu(out, name='relu')
            return out

    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def roi_pool(self, input, pooled_height, pooled_width, spatial_scale, name):
        # only use the first input
        if isinstance(input[0], tuple):
            input[0] = input[0][0]

        if isinstance(input[1], tuple):
            input[1] = input[1][0]

        print input
        return roi_pool_op.roi_pool(input[0], input[1],
                                    pooled_height,
                                    pooled_width,
                                    spatial_scale,
                                    name=name)[0]

    @layer
    def proposal_layer(self, input, _feat_stride, anchor_scales, anchor_ratios, cfg_key, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]
        return tf.reshape(tf.py_func(proposal_layer_py,
                                     [input[0], input[1], input[2], cfg_key, _feat_stride, anchor_scales,
                                      anchor_ratios],
                                     [tf.float32]), [-1, 5], name=name)

    @layer
    def anchor_target_layer(self, input, _feat_stride, anchor_scales, anchor_ratios, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]

        with tf.variable_scope(name) as scope:
            rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
                tf.py_func(anchor_target_layer_py,
                           [input[0], input[1], input[2], input[3], _feat_stride, anchor_scales, anchor_ratios],
                           [tf.float32, tf.float32, tf.float32, tf.float32])

            rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels, tf.int32), name='rpn_labels')
            rpn_bbox_targets = tf.convert_to_tensor(rpn_bbox_targets, name='rpn_bbox_targets')
            rpn_bbox_inside_weights = tf.convert_to_tensor(rpn_bbox_inside_weights, name='rpn_bbox_inside_weights')
            rpn_bbox_outside_weights = tf.convert_to_tensor(rpn_bbox_outside_weights, name='rpn_bbox_outside_weights')

            self._anchor_targets['rpn_labels'] = rpn_labels
            self._anchor_targets['rpn_bbox_targets'] = rpn_bbox_targets
            self._anchor_targets['rpn_bbox_inside_weights'] = rpn_bbox_inside_weights
            self._anchor_targets['rpn_bbox_outside_weights'] = rpn_bbox_outside_weights

            self._score_summaries.update(self._anchor_targets)

            return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights

    @layer
    def proposal_target_layer(self, input, classes, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]
        with tf.variable_scope(name) as scope:
            rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func(proposal_target_layer_py,
                                                                                               [input[0], input[1],
                                                                                                classes],
                                                                                               [tf.float32, tf.float32,
                                                                                                tf.float32, tf.float32,
                                                                                                tf.float32])

            rois = tf.reshape(rois, [-1, 5], name='rois')
            labels = tf.convert_to_tensor(tf.cast(labels, tf.int32), name='labels')
            bbox_targets = tf.convert_to_tensor(bbox_targets, name='bbox_targets')
            bbox_inside_weights = tf.convert_to_tensor(bbox_inside_weights, name='bbox_inside_weights')
            bbox_outside_weights = tf.convert_to_tensor(bbox_outside_weights, name='bbox_outside_weights')

            self._proposal_targets['rois'] = rois
            self._proposal_targets['labels'] = labels
            self._proposal_targets['bbox_targets'] = bbox_targets
            self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
            self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights

            self._score_summaries.update(self._proposal_targets)

            return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

    @layer
    def proposal_target_layer_3d(self, input, classes, name):
        if isinstance(input[0], tuple):
            input[0] = input[0][0]
        with tf.variable_scope(name) as scope:
            rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func(
                proposal_target_layer_3d_py,
                [input[0], input[1], input[2],
                 classes],
                [tf.float32, tf.float32,
                 tf.float32, tf.float32,
                 tf.float32])

            rois = tf.reshape(rois, [-1, 5], name='rois')
            labels = tf.convert_to_tensor(tf.cast(labels, tf.int32), name='labels')
            bbox_targets = tf.convert_to_tensor(bbox_targets, name='bbox_targets')
            bbox_inside_weights = tf.convert_to_tensor(bbox_inside_weights, name='bbox_inside_weights')
            bbox_outside_weights = tf.convert_to_tensor(bbox_outside_weights, name='bbox_outside_weights')

            self._proposal_targets['rois'] = rois
            self._proposal_targets['labels'] = labels
            self._proposal_targets['bbox_targets'] = bbox_targets
            self._proposal_targets['bbox_inside_weights'] = bbox_inside_weights
            self._proposal_targets['bbox_outside_weights'] = bbox_outside_weights

            self._score_summaries.update(self._proposal_targets)

            return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

    @layer
    def reshape_layer(self, input, d, name):
        input_shape = tf.shape(input)
        if name == 'rpn_cls_prob_reshape':
            return tf.transpose(tf.reshape(tf.transpose(input, [0, 3, 1, 2]), [input_shape[0],
                                                                               int(d), tf.cast(
                    tf.cast(input_shape[1], tf.float32) / tf.cast(d, tf.float32) * tf.cast(input_shape[3], tf.float32),
                    tf.int32), input_shape[2]]), [0, 2, 3, 1], name=name)
        else:
            return tf.transpose(tf.reshape(tf.transpose(input, [0, 3, 1, 2]), [input_shape[0],
                                                                               int(d), tf.cast(
                    tf.cast(input_shape[1], tf.float32) * (
                        tf.cast(input_shape[3], tf.float32) / tf.cast(d, tf.float32)), tf.int32), input_shape[2]]),
                                [0, 2, 3, 1], name=name)

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(concat_dim=axis, values=inputs, name=name)

    @layer
    def fc(self, input, num_out, name, trainable=True, relu=True, batch_norm=True, phase=False, bn_scale=True):
        with tf.variable_scope(name) as scope:
            # only use the first input
            if isinstance(input, tuple):
                input = input[0]

            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(tf.transpose(input, [0, 3, 1, 2]), [-1, dim])
            else:
                feed_in, dim = (input, int(input_shape[-1]))
            if batch_norm:
                normalizer_fn = layers.batch_norm
                normalizer_prms = {'is_training': phase, 'scale': bn_scale, 'fused': False}
            else:
                normalizer_prms = {}
                normalizer_fn = None
            if relu:
                activation_fn = tf.nn.relu
            else:
                activation_fn = None
            out = layers.fully_connected(feed_in, num_out, activation_fn=activation_fn, scope='fc',
                                         weights_regularizer=layers.l2_regularizer(cfg.TRAIN.REGULARIZER),
                                         weights_initializer=layers.xavier_initializer(uniform=False),
                                         biases_regularizer=None,
                                         biases_initializer=tf.constant_initializer(0.),
                                         normalizer_fn=normalizer_fn,
                                         normalizer_params=normalizer_prms)
            return out

    @layer
    def softmax(self, input, name):
        input_shape = tf.shape(input)
        if name == 'rpn_cls_prob':
            return tf.reshape(tf.nn.softmax(tf.reshape(input, [-1, input_shape[3]])),
                              [-1, input_shape[1], input_shape[2], input_shape[3]], name=name)
        else:
            return tf.nn.softmax(input, name=name)

    @layer
    def dropout(self, input, keep_prob, name):
        return tf.nn.dropout(input, keep_prob, name=name)

    def _add_act_summary(self, tensor):
        tf.summary.histogram('ACT/' + tensor.op.name + '/activations', tensor)
        tf.summary.scalar('ACT/' + tensor.op.name + '/zero_fraction',
                          tf.nn.zero_fraction(tensor))

    def _add_score_summary(self, key, tensor):
        tf.summary.histogram('SCORE/' + tensor.op.name + '/' + key + '/scores', tensor)

    def _add_train_summary(self, var):
        tf.summary.histogram('TRAIN/' + var.op.name, var)

    def _modified_smooth_l1(self, sigma, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights):
        """
            ResultLoss = outside_weights * SmoothL1(inside_weights * (bbox_pred - bbox_targets))
            SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                          |x| - 0.5 / sigma^2,    otherwise
        """
        sigma2 = sigma * sigma

        inside_mul = tf.multiply(bbox_inside_weights, tf.subtract(bbox_pred, bbox_targets))

        smooth_l1_sign = tf.cast(tf.less(tf.abs(inside_mul), 1.0 / sigma2), tf.float32)
        smooth_l1_option1 = tf.multiply(tf.multiply(inside_mul, inside_mul), 0.5 * sigma2)
        smooth_l1_option2 = tf.subtract(tf.abs(inside_mul), 0.5 / sigma2)
        smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
                                  tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))

        outside_mul = tf.multiply(bbox_outside_weights, smooth_l1_result)

        return outside_mul

    def add_losses(self):
        # RPN
        # classification loss
        rpn_cls_score = tf.reshape(self.get_output('rpn_cls_score_reshape'), [-1, 2])
        # convert to vector
        rpn_label = tf.reshape(self.get_output('rpn-data')[0], [-1])
        # filter rpn cls scores and labels where their corresponding labels have an ignore label
        labels_not_ignored_idxs = tf.where(tf.not_equal(rpn_label, -1))
        rpn_cls_score2 = tf.reshape(tf.gather(rpn_cls_score, labels_not_ignored_idxs), [-1, 2])
        rpn_label = tf.reshape(tf.gather(rpn_label, labels_not_ignored_idxs), [-1])
        # calculate loss
        rpn_cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score2, labels=rpn_label))

        # bounding box regression L1 loss
        rpn_bbox_pred = self.get_output('rpn_bbox_pred')
        rpn_bbox_targets = tf.transpose(self.get_output('rpn-data')[1], [0, 2, 3, 1])
        rpn_bbox_inside_weights = tf.transpose(self.get_output('rpn-data')[2], [0, 2, 3, 1])
        rpn_bbox_outside_weights = tf.transpose(self.get_output('rpn-data')[3], [0, 2, 3, 1])

        rpn_smooth_l1 = self._modified_smooth_l1(3.0, rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                 rpn_bbox_outside_weights)
        rpn_loss_box = tf.multiply(tf.reduce_mean(tf.reduce_sum(rpn_smooth_l1, reduction_indices=[1, 2, 3])), 10)

        # R-CNN
        # classification loss
        cls_score = self.get_output('cls_score')
        label = tf.reshape(self.get_output('roi-data')[1], [-1])
        cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score, labels=label))

        # bounding box regression L1 loss
        bbox_pred = self.get_output('bbox_pred')
        bbox_targets = self.get_output('roi-data')[2]
        bbox_inside_weights = self.get_output('roi-data')[3]
        bbox_outside_weights = self.get_output('roi-data')[4]

        smooth_l1 = self._modified_smooth_l1(1.0, bbox_pred, bbox_targets, bbox_inside_weights,
                                             bbox_outside_weights)
        loss_box = tf.multiply(tf.reduce_mean(tf.reduce_sum(smooth_l1, reduction_indices=[1])), 1)

        rpn_cross_entropy.set_shape([])
        rpn_loss_box.set_shape([])
        cross_entropy.set_shape([])
        loss_box.set_shape([])

        self._losses['rpn_cross_entropy'] = rpn_cross_entropy
        self._losses['rpn_loss_box'] = rpn_loss_box
        self._losses['cross_entropy'] = cross_entropy
        self._losses['loss_box'] = loss_box

        # final loss
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss = rpn_cross_entropy + rpn_loss_box + cross_entropy + loss_box + sum(reg_losses)
        self._losses['total_loss'] = loss

        # create smoothed shadow versions of losses
        ema_decay = 0.99
        ema = tf.train.ExponentialMovingAverage(ema_decay)
        self._ema_ops['losses'] = ema.apply([rpn_cross_entropy, rpn_loss_box, cross_entropy, loss_box, loss])
        self._losses['rpn_cross_entropy_smoothed'] = ema.average(rpn_cross_entropy)
        self._losses['rpn_loss_box_smoothed'] = ema.average(rpn_loss_box)
        self._losses['cross_entropy_smoothed'] = ema.average(cross_entropy)
        self._losses['loss_box_smoothed'] = ema.average(loss_box)
        self._losses['total_loss_smoothed'] = ema.average(loss)

        self.train_dict['loss_op'] = self._losses['total_loss']

        self._value_summaries.update(self._losses)

    def _add_pred_boxes_calc(self):
        rois = tf.slice(self._predictions['rois'], [0, 1], [-1, -1])
        bbox_pred = self._predictions['bbox_pred']
        labels = self._proposal_targets['labels']
        boxes_3d = tf.py_func(bbox3d_transform_inv, [rois, bbox_pred], tf.float32)
        boxes_3d_reduced = tf.py_func(bbox_reduce, [boxes_3d, labels], tf.float32)
        boxes_3d_reduced.set_shape([None, 7])
        boxes_2d, _, _ = tf.py_func(get_3d_box_top_view, [boxes_3d_reduced], [tf.float64, tf.int64, tf.float32])
        boxes_2d = tf.cast(boxes_2d, tf.float32)
        boxes_2d.set_shape([None, 4])
        self._predictions['output_boxes'] = boxes_2d

    def _add_fg_rois(self):
        rois = tf.slice(self._predictions['rois'], [0, 1], [-1, -1])
        labels = self._proposal_targets['labels']
        fg_idxs = tf.where(tf.not_equal(labels, 0))
        self._proposal_targets['fg_rois'] = tf.reshape(tf.gather(rois, fg_idxs), [-1, 4])

    def create_summaries(self):
        small_summaries = []
        val_summaries = []
        for var in tf.trainable_variables():
            self._train_summaries.append(var)
        with tf.device("/cpu:0"):
            self._add_image_summary(self.data, self.gt_boxes, 'ground_truth')
            self._add_image_summary(self.data, self._predictions['output_boxes'], 'predictions')
            self._add_image_summary(self.data, self._proposal_targets['fg_rois'], 'rois')
            small_summaries.append(tf.summary.scalar('Learning_rate', self.train_dict['lr']))
            for key, val in self._value_summaries.items():
                summary = tf.summary.scalar(key, val)
                val_summaries.append(summary)
                small_summaries.append(summary)
            for key, var in self._score_summaries.items():
                self._add_score_summary(key, var)
            for var in self._act_summaries:
                self._add_act_summary(var)
            for var in self._train_summaries:
                self._add_train_summary(var)

            self.train_dict['fast_summary_op'] = tf.summary.merge(small_summaries)
            self.train_dict['slow_summary_op'] = tf.summary.merge_all()

    def _add_image_summary(self, data, boxes, tag):
        # add back mean
        image = tf.slice(data, [0, 0, 0, 0], [1, -1, -1, 1])
        # dims for normalization
        width = tf.to_float(tf.shape(image)[2])
        height = tf.to_float(tf.shape(image)[1])
        # from [x1, y1, x2, y2, cls] to normalized [y1, x1, y1, x1]
        cols = tf.unstack(boxes, axis=1)
        boxes = tf.stack([cols[1] / height,
                          cols[0] / width,
                          cols[3] / height,
                          cols[2] / width], axis=1)
        # add batch dimension (assume batch_size==1)
        assert image.get_shape()[0] == 1
        boxes = tf.expand_dims(boxes, dim=0)
        image = tf.image.draw_bounding_boxes(image, boxes)

        return tf.summary.image(tag, image)

    def _feed_dict(self, blobs, phase):
        feed_dict = {self.data: blobs['data'][0], self.im_info: blobs['im_info'],
                     self.bbox_pred_means: self._bbox_means, self.bbox_pred_stds: self._bbox_stds}
        if 'gt_boxes' in blobs:
            feed_dict[self.gt_boxes] = blobs['gt_boxes']
        if 'gt_boxes_3d' in blobs:
            feed_dict[self.gt_boxes_3d] = blobs['gt_boxes_3d']

        if phase == 'TRAIN':
            feed_dict[self.keep_prob] = cfg.TRAIN.DROPOUT_PROB
            feed_dict[self.is_training] = True
        else:
            feed_dict[self.keep_prob] = 1.
            feed_dict[self.is_training] = False

        return feed_dict

    def setup_training(self, means, stds):
        # add losses to graph
        self.add_losses()
        # optimizer and learning rate
        self.train_dict['global_step'] = tf.Variable(0, trainable=False)
        lr = self.train_dict['lr'] = tf.Variable(cfg.TRAIN.LEARNING_RATE, trainable=False)
        self.train_dict['last_update_lr_step'] = 0
        self._bbox_means = means.reshape((1, 28))
        self._bbox_stds = stds.reshape((1, 28))
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # for batch norm
        with tf.control_dependencies(update_ops):
            # Ensures that we execute the update_ops before performing the train_op
            loss_op = self.train_dict['loss_op']
            global_step = self.train_dict['global_step']
            if cfg.TRAIN.SOLVER == 'Momentum':
                momentum = cfg.TRAIN.MOMENTUM
                train_op = tf.train.MomentumOptimizer(lr, momentum).minimize(loss_op, global_step=global_step)
            elif cfg.TRAIN.SOLVER == 'Adam':
                beta1 = cfg.TRAIN.BETA1
                beta2 = cfg.TRAIN.BETA2
                train_op = tf.train.AdamOptimizer(lr, beta1, beta2).minimize(loss_op, global_step=global_step)
        self.create_summaries()

        with tf.control_dependencies([train_op]):
            self.train_dict['train_op'] = tf.group(self._ema_ops.values()[0])

    def train_step_no_return(self, sess, blobs):
        sess.run([self.train_dict['train_op']], self._feed_dict(blobs, 'TRAIN'))

    def train_step(self, sess, blobs):
        _, loss, loss_smoothed, summary = sess.run([self.train_dict['train_op'], self._losses['total_loss'],
                                                    self._losses['total_loss_smoothed'],
                                                    self.train_dict['fast_summary_op']],
                                                   self._feed_dict(blobs, 'TRAIN'))
        return loss, loss_smoothed, summary

    def train_step_summary(self, sess, blobs):
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        _, loss, loss_smoothed, summary = sess.run([self.train_dict['train_op'],
                                                    self._losses['total_loss'],
                                                    self._losses['total_loss_smoothed'],
                                                    self.train_dict['slow_summary_op']],
                                                   self._feed_dict(blobs, 'TRAIN'),
                                                   options=run_options,
                                                   run_metadata=run_metadata)
        return loss, loss_smoothed, summary, run_metadata

    def test(self, sess, blobs):
        loss = self.train_dict['loss_op']
        in_ops = self._predictions.values()
        in_ops.append(loss)
        predictions = sess.run(in_ops, self._feed_dict(blobs, 'TEST'))
        loss_val = predictions[-1]
        outputs_val = [predictions[self._predictions.keys().index(x)] for x in
                       ['cls_score', 'cls_prob', 'bbox_pred', 'rois']]
        precision, recall, error_rate = self.eval_detections(blobs, outputs_val, det_min_iou=0.5)
        return precision, recall, error_rate, loss_val

    def eval_detections(self, blobs, outputs_val, det_min_iou):
        """Returns a recall one hot array corresponding to ground truth objects with iou larger than det_min_iou with 
        detected objects. Returns a recall one hot array corresponding to classification correctness of each object. 
        Association for classification error is done using asso_min_iou"""
        # get image
        im = blobs['data'][0][0, :, :, 0]
        im[im == 0.] = im.min()
        # get bounding boxes of ground truth objects of current iteration
        gt_boxes_3d = blobs['gt_boxes_3d']
        gt_boxes_2d = np.vstack([get_3d_box_top_view(gt_boxes_3d[i, :-1])[0] for i in xrange(gt_boxes_3d.shape[0])])
        true_classes = gt_boxes_3d[:, -1]
        # get bounding boxes of detected objects of current iteration
        cls_score, cls_prob, bbox_deltas, rois = outputs_val[0], outputs_val[1], outputs_val[2], outputs_val[3]
        if rois.shape[0] == 0:
            return 0., 0., 1.
        boxes = rois[:, 1:5]
        pred_boxes_3d = bbox3d_transform_inv(boxes, bbox_deltas)
        classes = cls_score.argmax(axis=1)
        row_inds = np.arange(pred_boxes_3d.shape[0])[:, np.newaxis]
        col_inds = 7 * classes[:, np.newaxis] + np.arange(7)[np.newaxis, :]
        pred_boxes_reduced = pred_boxes_3d[row_inds, col_inds]
        pred_boxes_3d_top_view, keep_inds, _ = get_3d_box_top_view(pred_boxes_reduced)
        pred_boxes_3d_top_view_classified = pred_boxes_3d_top_view[classes[keep_inds] != 0]

        # calculate IOUs
        if pred_boxes_3d_top_view_classified.size > 0:
            ious = bbox_overlaps(pred_boxes_3d_top_view_classified.astype(np.float), gt_boxes_2d.astype(np.float))
            # find best corresponding detected objects for each ground truth object
            max_overlaps_gt = ious.max(axis=0)
            max_overlaps_det = ious.max(axis=1)
            max_overlaps_inds = ious.argmax(axis=0)
            gt_objects = true_classes.size
            error_rate = float(np.sum((true_classes != classes[max_overlaps_inds]) |
                                      (ious[max_overlaps_inds, np.arange(gt_objects)] == 0))) / gt_objects
            # calculate iou hit array
            recall = float((max_overlaps_gt > det_min_iou).sum()) / gt_boxes_3d.shape[0]
            precision = float((max_overlaps_det > det_min_iou).sum()) / pred_boxes_3d_top_view_classified.shape[0]
        else:
            error_rate = 1.
            recall = 0.
            precision = 0.

        return precision, recall, error_rate