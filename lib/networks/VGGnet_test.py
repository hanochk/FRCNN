import tensorflow as tf
from networks.network import Network
from fast_rcnn.config import cfg
import numpy as np

n_classes = 4
n_channels = cfg.NET.INPUT_CHANNELS
_feat_stride = cfg.NET.DOWNSAMPLE  # this is set according to the decimation of the feature extractor and cannot be set
anchor_scales = cfg.NET.RPN_ANCHOR_SCALES
anchor_ratios = cfg.NET.RPN_ANCHOR_RATIOS


class VGGnet_test(Network):
    def __init__(self, trainable=True, modality='point_cloud'):
        self._modality = modality
        if self._modality == 'point_cloud':
            chan = n_channels
        else:
            chan = 3

        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, chan])
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'data': self.data, 'im_info': self.im_info})
        self.trainable = trainable
        self._modality = modality
        self.setup()

    def get_layer_output(self, sess, layer_name, image, im_info=np.array([[512, 256, 1]])):
        feed_dict = {self.data: image, self.im_info: im_info}
        with sess.as_default():
            return self.get_output(layer_name).eval(feed_dict)

    def setup(self):
        phase = False
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
         #.max_pool(2, 2, 2, 2, padding='VALID', name='pool4')
         .conv(3, 3, 256, 1, 1, name='conv5_1', phase=phase)
         .conv(3, 3, 256, 1, 1, name='conv5_2', phase=phase)
         .conv(3, 3, 256, 1, 1, name='conv5_3', phase=phase))

        anchor_num = len(anchor_scales) * len(anchor_ratios)
        (self.feed('conv5_3')
         .conv(3, 3, 512, 1, 1, name='rpn_conv/3x3', phase=phase)
         .conv(1, 1, anchor_num * 2, 1, 1, padding='VALID', relu=False, name='rpn_cls_score', phase=phase))

        (self.feed('rpn_conv/3x3')
         .conv(1, 1, anchor_num * 4, 1, 1, padding='VALID', relu=False, name='rpn_bbox_pred', phase=phase))

        (self.feed('rpn_cls_score')
         .reshape_layer(2, name='rpn_cls_score_reshape')
         .softmax(name='rpn_cls_prob'))

        (self.feed('rpn_cls_prob')
         .reshape_layer(anchor_num * 2, name='rpn_cls_prob_reshape'))

        (self.feed('rpn_cls_prob_reshape', 'rpn_bbox_pred', 'im_info')
         .proposal_layer(_feat_stride, anchor_scales, anchor_ratios, 'TEST', name='rois'))

        (self.feed('conv5_3', 'rois')
         .roi_pool(7, 7, 1.0 / _feat_stride, name='pool_5')
         .fc(4096, name='fc6', phase=phase)
         .fc(4096, name='fc7', phase=phase)
         .fc(4096, name='fc8', phase=phase)
         .fc(n_classes, relu=False, name='cls_score', phase=phase)
         .softmax(name='cls_prob'))

        if self._modality == 'point_cloud':
            (self.feed('fc8')
             .fc(n_classes * 7, relu=False, name='bbox_pred', phase=phase))
        else:
            (self.feed('fc8')
             .fc(n_classes * 4, relu=False, name='bbox_pred', phase=phase))
