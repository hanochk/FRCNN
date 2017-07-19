from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg,cfg_from_file, get_output_dir
from datasets.factory import get_imdb
from networks.factory import get_network
import pprint
import datetime
import numpy as np
import tensorflow as tf
import cPickle

models = ['data/pretrain_model/VGG_imagenet.npy', 'output/faster_rcnn_end2end/kitti3d_trainval_170330_165345/VGGnet_fast_rcnn_iter_25000.ckpt']
n_trials = 50
max_iters = 15000
mu_range = [-6, -4]
momentum_range = [0.5, 0.99]
trial_vec = []

experiment = 'exp1'

for trial in xrange(n_trials):
    tf.reset_default_graph()
    params = {}
    params['learning_rate'] = 10 ** np.random.uniform(mu_range[0], mu_range[1])
    params['momentum'] = np.random.uniform(momentum_range[0], momentum_range[1])
    params['pretrained_model'] = np.random.choice(models, 1)[0]
    print 'Starting trial {}, params:'.format(trial)
    print params
    try:
        device = 'gpu'
        device_id = 0
        cfg_file = 'experiments/cfgs/faster_rcnn_end2end.yml'
        imdb_name = 'kitti3d_trainval'
        network_name = 'VGGnet_train'
        #pretrained_model = 'data/pretrain_model/VGG_imagenet.npy'
        #pretrained_model = 'output/faster_rcnn_end2end/kitti3d_trainval_170330_165345/VGGnet_fast_rcnn_iter_25000.ckpt'
        pretrained_model = params['pretrained_model']

        #max_iters = 300000
        modality = 'point_cloud'

        if cfg_file is not None:
            cfg_from_file(cfg_file)
        cfg.TRAIN.LEARNING_RATE = params['learning_rate']
        cfg.TRAIN.MOMENTUM = params['momentum']
        print('Using config:')
        pprint.pprint(cfg)

        imdb = get_imdb(imdb_name)
        print 'Loaded dataset `{:s}` for training'.format(imdb.name)
        roidb = get_training_roidb(imdb)

        output_dir = get_output_dir(imdb, None) + experiment + '_{}'.format(datetime.datetime.now().strftime("%y%m%d_%H%M%S"))
        print 'Output will be saved to `{:s}`'.format(output_dir)

        device_name = '/{}:{:d}'.format(device, device_id)
        print device_name

        network = get_network(network_name)
        print 'Use network `{:s}` in training'.format(network_name)

        results = dict()
        results['loss'] = train_net(network, imdb, roidb, output_dir, experiment,
                  pretrained_model=pretrained_model,
                  max_iters=max_iters,
                  modality=modality)
        trial_vec.append({'results': results, 'params': params})
    except:
        pass
    results_filename = 'results.pkl'
    with open(results_filename, 'wb') as f:
        cPickle.dump(trial_vec, f, cPickle.HIGHEST_PROTOCOL)
