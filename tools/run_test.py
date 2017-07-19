from fast_rcnn.test import test_net
from fast_rcnn.config import cfg,cfg_from_file, get_output_dir
from datasets.factory import get_imdb
from networks.factory import get_network
import pprint
import datetime
import os
import tensorflow as tf

device = 'gpu'
device_id = 0
cfg_file = 'experiments/cfgs/faster_rcnn_end2end.yml'
imdb_name = 'kitti3d_trainval'
network_name = 'VGGnet_test'
model = None
# pretrained_model = 'data/pretrain_model/VGG_imagenet.npy'
model = \
 '/fastdata/outputs/default/kitti3d_trainval_170425_092709/VGGnet_fast_rcnn_iter_5188.ckpt'
max_iters = 1000000
modality = 'point_cloud'

#if cfg_file is not None:
#    cfg_from_file(cfg_file)

print('Using config:')
pprint.pprint(cfg)

weights_filename = os.path.splitext(os.path.basename(model))[0]

imdb = get_imdb(imdb_name)

device_name = '/{}:{:d}'.format(device, device_id)
print device_name

network = get_network(network_name)
print 'Use network `{:s}` in testing'.format(network_name)

if device == 'gpu':
    cfg.USE_GPU_NMS = True
    cfg.GPU_ID = device_id
else:
    cfg.USE_GPU_NMS = False

# find latest checkpoint
file_name = model
# file_list = [file for file in os.listdir(folder) if ('.ckpt.data' in file) or (file.split('.')[-1] == 'ckpt')]
# for file in os.listdir(folder):
#     print file.split('.')[-1]
# print 'file list = {}'.format(file_list)
# ckpt_iter = [int(name.split('_')[-1].split('.')[0]) for name in file_list]
#
# file_name = file_list[np.argmax(ckpt_iter)]
# if file_name.split('.')[-1] != 'ckpt':
#     file_name = '.'.join(file_name.split('.')[:-1])
print 'FILENAME={}'.format(file_name)
# start a session
saver = tf.train.Saver()
config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(allow_growth=True)

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
saver.restore(sess, file_name)
print ('Loading model weights from {:s}').format(model)

test_net(sess, network, imdb, weights_filename, vis=True, thresh=0.999)
