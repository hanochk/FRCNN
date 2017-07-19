from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg,cfg_from_file, get_output_dir
from datasets.factory import get_imdb
from networks.factory import get_network
import pprint
import datetime
import os

device = 'gpu'
device_id = 0
cfg_file = 'experiments/cfgs/faster_rcnn_end2end.yml'
imdb_name = 'kitti3d_train'
val_imdb_name = 'kitti3d_val'
network_name = 'VGGnet_train'
pretrained_model = None
experiment_id = None
# pretrained_model = 'data/pretrain_model/VGG_imagenet.npy'
# pretrained_model = \
#  '/fastdata/outputs/default/kitti3d_train_170501_213500/VGGnet_fast_rcnn_iter_520000.ckpt'
max_iters = 1000000
modality = 'point_cloud'

#if cfg_file is not None:
#    cfg_from_file(cfg_file)

print('Using config:')
pprint.pprint(cfg)

imdb = get_imdb(imdb_name)
print 'Loaded dataset `{:s}` for training'.format(imdb.name)
roidb = get_training_roidb(imdb)

if val_imdb_name is not None:
    imdb_val = get_imdb(val_imdb_name)
    print 'Loaded dataset `{:s}` for training'.format(imdb.name)
    roidb_val = get_training_roidb(imdb_val)
else:
    imdb_val = None
    roidb_val = None

output_dir = get_output_dir(imdb, None) + '_{}'.format(datetime.datetime.now().strftime("%y%m%d_%H%M%S"))
print 'Output will be saved to `{:s}`'.format(output_dir)

device_name = '/{}:{:d}'.format(device, device_id)
print device_name

network = get_network(network_name)
print 'Use network `{:s}` in training'.format(network_name)

train_net(network, imdb, roidb, imdb_val, roidb_val, output_dir,
          pretrained_model=pretrained_model,
          max_iters=max_iters,
          modality=modality)

