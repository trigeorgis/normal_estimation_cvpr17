import tensorflow as tf
import numpy as np
import network
import losses
import data_provider

from tensorflow.python.platform import tf_logging as logging
from pathlib import Path

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('initial_learning_rate', 0.0001,
                          '''Initial learning rate.''')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 5.0,
                          '''Epochs after which learning rate decays.''')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.97,
                          '''Learning rate decay factor.''')
tf.app.flags.DEFINE_integer('batch_size', 4, '''The batch size to use.''')
tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
                            '''How many preprocess threads to use.''')
tf.app.flags.DEFINE_string('train_dir', 'ckpt/train',
                           '''Directory where to write event logs '''
                           '''and checkpoint.''')
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '',
                           '''If specified, restore this pretrained model '''
                           '''before beginning any training.''')
tf.app.flags.DEFINE_string(
    'pretrained_resnet_checkpoint_path', '',
    '''If specified, restore this pretrained resnet '''
    '''before beginning any training.'''
    '''This restores only the weights of the resnet model''')
tf.app.flags.DEFINE_integer('max_steps', 100000,
                            '''Number of batches to run.''')
tf.app.flags.DEFINE_string('train_device', '/gpu:0',
                           '''Device to train with.''')

# The decay to use for the moving average.
MOVING_AVERAGE_DECAY = 0.9999


def restore_resnet(path):
    def name_in_checkpoint(var):
        name = '/'.join(var.name.split('/')[2:])
        name = name.split(':')[0]
        if 'Adam' in name:
            return None

        return name
    
    variables_to_restore = slim.get_variables_to_restore(
        include=["net"])
    
    variables_to_restore = {name_in_checkpoint(var): var
                            for var in variables_to_restore if name_in_checkpoint(var) is not None}

    init_fn = slim.assign_from_checkpoint_fn(
                    path,
                    variables_to_restore, ignore_missing_vars=True)
    
    return init_fn

def train():
    g = tf.Graph()
    with g.as_default():
        # Load datasets.
        names = ['PhotofaceNormals', 'SyntheticNormals']
        provider = data_provider.DatasetMixer(
            names, batch_size=FLAGS.batch_size, densities=(10, 1))
        images, normals, mask = provider.get()
        
        # Define model graph.
        with tf.variable_scope('net'):
            with slim.arg_scope([slim.batch_norm, slim.layers.dropout],
                                is_training=True):
                scales = [1, 2]
                prediction, pyramid = network.multiscale_nrm_net(images, scales=scales)

        # Add a smooth l1 loss to every scale and the combined output.
        for net, level_name in zip([prediction] + pyramid, ['pred'] + scales):
            loss = losses.smooth_l1(net, normals, mask)
            tf.scalar_summary('losses/loss at {}'.format(level_name), loss)

        total_loss = slim.losses.get_total_loss()
        tf.scalar_summary('losses/total loss', total_loss)
        tf.image_summary('images', images)
        tf.image_summary('normals', normals)
        tf.image_summary('predictions', prediction)

        optimizer = tf.train.AdamOptimizer(FLAGS.initial_learning_rate)

    config = tf.ConfigProto(inter_op_parallelism_threads=2)
    with tf.Session(graph=g, config=config) as sess:

        init_fn = None
        if FLAGS.pretrained_resnet_checkpoint_path:
            init_fn = restore_resnet(FLAGS.pretrained_resnet_checkpoint_path)

        if FLAGS.pretrained_model_checkpoint_path:
            print('Loading whole model...')
            variables_to_restore = slim.get_model_variables()
            init_fn = slim.assign_from_checkpoint_fn(
                    FLAGS.pretrained_model_checkpoint_path,
                    variables_to_restore, ignore_missing_vars=True)

        train_op = slim.learning.create_train_op(total_loss,
                                                 optimizer,
                                                 summarize_gradients=True)

        logging.set_verbosity(1)
        
        slim.learning.train(train_op,
                            FLAGS.train_dir,
                            save_summaries_secs=60,
                            init_fn=init_fn,
                            save_interval_secs=600)


if __name__ == '__main__':
    train()
