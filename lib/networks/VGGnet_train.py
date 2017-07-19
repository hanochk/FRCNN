import tensorflow as tf
from networks.network import Network
from fast_rcnn.config import cfg
import numpy as np

# define

n_channels = cfg.NET.INPUT_CHANNELS  # number of input channels for the network
n_classes = 4
_feat_stride = cfg.NET.DOWNSAMPLE  # this is set according to the decimation of the feature extractor and cannot be set
anchor_scales = cfg.NET.RPN_ANCHOR_SCALES
anchor_ratios = cfg.NET.RPN_ANCHOR_RATIOS


class VGGnet_train(Network):
    def __init__(self, trainable=True, modality='point_cloud'):
        self._modality = modality
        if self._modality == 'point_cloud':
            chan = n_channels
        else:
            chan = 3
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, chan])
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
        self.gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
        self.keep_prob = tf.placeholder(tf.float32)
        self.is_training = tf.placeholder(tf.bool)
        self.bbox_pred_means = tf.placeholder(tf.float32, shape=[1, 28])
        self.bbox_pred_stds = tf.placeholder(tf.float32, shape=[1, 28])
        self.layers = {'data': self.data, 'im_info': self.im_info, 'gt_boxes': self.gt_boxes,
                       'keep_prob': self.keep_prob, 'is_training': self.is_training}
        if self._modality == 'point_cloud':
            self.gt_boxes_3d = tf.placeholder(tf.float32, shape=[None, 8])
            self.layers['gt_boxes_3d'] = self.gt_boxes_3d
        self.trainable = trainable
        Network.__init__(self, self.layers, trainable)
        self.setup()

        # create ops and placeholders for bbox normalization process
        with tf.variable_scope('bbox_pred/fc', reuse=True):
            weights = tf.get_variable("weights")
            biases = tf.get_variable("biases")

            self.bbox_weights = tf.placeholder(weights.dtype, shape=weights.get_shape())
            self.bbox_biases = tf.placeholder(biases.dtype, shape=biases.get_shape())

            self.bbox_weights_assign = weights.assign(self.bbox_weights)
            self.bbox_bias_assign = biases.assign(self.bbox_biases)

    def get_layer_output(self, layer_name, image, im_info=np.array([512, 256, 1])):
        feed_dict = {self.data: image, self.data: im_info}
        self.get_output(layer_name).eval(feed_dict)

    def setup(self):
        phase = self.is_training
        (self.feed('data')
         .conv(3, 3, 32, 1, 1, name='conv1_1', phase=phase)
         .conv(3, 3, 32, 1, 1, name='conv1_2', phase=phase)

         .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
         .conv(3, 3, 64, 1, 1, name='conv2_1', phase=phase)
         .conv(3, 3, 64, 1, 1, name='conv2_2', phase=phase)

         .max_pool(2, 2, 2, 2, padding='VALID', name='pool2')
         .conv(3, 3, 128, 1, 1, name='conv3_1', phase=phase)
         .conv(3, 3, 128, 1, 1, name='conv3_2', phase=phase)
         .conv(3, 3, 128, 1, 1, name='conv3_3', phase=phase)

         .max_pool(2, 2, 2, 2, padding='VALID', name='pool3')
         .conv(3, 3, 256, 1, 1, name='conv4_1', phase=phase)
         .conv(3, 3, 256, 1, 1, name='conv4_2', phase=phase)
         .conv(3, 3, 256, 1, 1, name='conv4_3', phase=phase)

         .conv(3, 3, 256, 1, 1, name='conv5_1', phase=phase)
         .conv(3, 3, 256, 1, 1, name='conv5_2', phase=phase)
         .conv(3, 3, 256, 1, 1, name='conv5_3', phase=phase)
         .deconv(3, 3, 256, 2, 2, name='deconv5_4', phase=phase))
        # ========= RPN ============
        anchor_num = len(anchor_scales) * len(anchor_ratios)
        (self.feed('deconv5_4')
         .conv(3, 3, 512, 1, 1, name='rpn_conv/3x3', phase=phase)
         .conv(1, 1, anchor_num * 2, 1, 1, padding='VALID', relu=False, name='rpn_cls_score', phase=phase))

        (self.feed('rpn_cls_score', 'gt_boxes', 'im_info', 'data')
         .anchor_target_layer(_feat_stride, anchor_scales, anchor_ratios, name='rpn-data'))

        # Loss of rpn_cls & rpn_boxes

        (self.feed('rpn_conv/3x3')
         .conv(1, 1, anchor_num * 4, 1, 1, padding='VALID', relu=False, name='rpn_bbox_pred', phase=phase,
               batch_norm=True))

        # ========= RoI Proposal ============
        (self.feed('rpn_cls_score')
         .reshape_layer(2, name='rpn_cls_score_reshape')
         .softmax(name='rpn_cls_prob'))

        (self.feed('rpn_cls_prob')
         .reshape_layer(anchor_num * 2, name='rpn_cls_prob_reshape'))

        (self.feed('rpn_cls_prob_reshape', 'rpn_bbox_pred', 'im_info')
         .proposal_layer(_feat_stride, anchor_scales, anchor_ratios, 'TRAIN', name='rpn_rois'))

        # convert gt boxes to rois created by rpn
        if self._modality == 'point_cloud':
            (self.feed('rpn_rois', 'gt_boxes', 'gt_boxes_3d')
             .proposal_target_layer_3d(n_classes, name='roi-data'))
        else:
            (self.feed('rpn_rois', 'gt_boxes')
             .proposal_target_layer(n_classes, name='roi-data'))

        # ========= RCNN ============
        (self.feed('deconv5_4', 'roi-data')
         .roi_pool(7, 7, 1.0 / _feat_stride, name='pool_5')
         .dropout(keep_prob=self.keep_prob, name='drop5')
         .fc(4096, name='fc6', batch_norm=False, phase=phase)
         .dropout(keep_prob=self.keep_prob, name='drop6')
         .fc(4096, name='fc7', batch_norm=False, phase=phase)
         .dropout(keep_prob=self.keep_prob, name='drop7')
         .fc(4096, name='fc8', batch_norm=False, phase=phase)
         .dropout(keep_prob=self.keep_prob, name='drop8')
         .fc(n_classes, relu=False, name='cls_score', batch_norm=True, phase=phase)
         .softmax(name='cls_prob'))
        if self._modality == 'point_cloud':
            (self.feed('drop8')
             .fc(n_classes * 7, relu=False, name='bbox_pred', batch_norm=False, phase=phase))
        else:
            (self.feed('drop8')
             .fc(n_classes * 4, relu=False, name='bbox_pred', batch_norm=False, phase=phase))


        self._predictions["rpn_cls_score"] = self.get_output('rpn_cls_score')
        self._predictions["rpn_cls_score_reshape"] = self.get_output('rpn_cls_score_reshape')
        self._predictions["rpn_cls_prob"] = self.get_output('rpn_cls_prob')
        self._predictions["rpn_bbox_pred"] = self.get_output('rpn_bbox_pred')
        self._predictions["cls_score"] = self.get_output('cls_score')
        self._predictions["cls_prob"] = self.get_output('cls_prob')
        self._predictions["bbox_pred"] = self.get_output('bbox_pred')
        self._predictions["rois"] = self.get_output('roi-data')[0]
        self._add_pred_boxes_calc()
        self._add_fg_rois()

        bbox_pred = self._predictions['bbox_pred']
        mean = self.bbox_pred_means
        std = self.bbox_pred_stds

        self._predictions['bbox_pred'] = tf.add(tf.multiply(bbox_pred, std), mean)