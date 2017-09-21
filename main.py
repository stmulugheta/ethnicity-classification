from model import ZGLFace

import numpy as np
import tensorflow as tf

import pprint
import os

flags = tf.app.flags
flags.DEFINE_integer("epoch", 1500, "Number of epoch [10]")
flags.DEFINE_integer("batch_size", 10, "The size of batch images [10]")
flags.DEFINE_integer("image_size", 64, "The size of image to use [64]")
flags.DEFINE_integer("label_num", 3, "The number of labels to use [3]")
flags.DEFINE_float("learning_rate", 1e-3, "The learning rate of gradient descent algorithm [1e-4]")
flags.DEFINE_integer("channel_num", 1, "Dimension of image color. [1]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [True]")
FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()
def main(_):
  pp.pprint(flags.FLAGS.__flags)

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.allow_soft_placement = True
  config.log_device_placement = True

  with tf.Session(config=config) as sess:
      zglface = ZGLFace(sess, 
                      image_size=FLAGS.image_size, 
                      label_num=FLAGS.label_num, 
                      batch_size=FLAGS.batch_size,
                      channel_num=FLAGS.channel_num, 
                      checkpoint_dir=FLAGS.checkpoint_dir)
      zglface.train(FLAGS)
    
if __name__ == '__main__':
    tf.app.run()