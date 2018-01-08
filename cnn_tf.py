# This file was created for the CS221 project.
# Julie Chang, Cynthia Hua

# Imports
import tensorflow as tf
import numpy as np
import pandas as pd
import rnn

# define the CNN encoder model
def cnn_model(features):
  """Model function for CNN."""

  # Input Layer [batch_size, image_width, image_height, channels]
  input_layer = tf.reshape(features, [-1, 48, 48, 1])

  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=4,
      kernel_size=[2, 2],
      strides=(2,2),
      padding="same",
      activation=tf.nn.relu)
  # output: [batch_size, 24, 24, 4]

  # Convolutional Layer #2 
  conv2 = tf.layers.conv2d(
      inputs=conv1,
      filters=4,
      kernel_size=[2, 2],
      padding="same",
      activation=tf.nn.relu)
  # output: [batch_size, 24, 24, 4]

  # Convolutional Layer #3
  conv3 = tf.layers.conv2d(
      inputs=conv2,
      filters=8,
      strides=(2,2),
      kernel_size=[2, 2],
      padding="same",
      activation=tf.nn.relu)
  # output: [batch_size, 12, 12, 8]
    
  # Convolutional Layer #4
  conv4 = tf.layers.conv2d(
      inputs=conv3,
      filters=8,
      strides=(2,2),
      kernel_size=[2, 2],
      padding="same",
      activation=tf.nn.relu)
  # output: [batch_size, 6, 6, 8]

  # Convolutional Layer #5
  conv5 = tf.layers.conv2d(
      inputs=conv4,
      filters=8,
      strides=(1,1),
      kernel_size=[2, 2],
      padding="same",
      activation=tf.nn.relu)
  # output: [batch_size, 6, 6, 8]

  # flatten our feature map to shape [batch_size, features], so that our tensor has only two dimensions:
  y_out = tf.reshape(conv5, [-1, 6 * 6 * 8])
  
  # return final layer
  return y_out

# This duplicates a function from the RNN model
# Returns W*x + b
def super_linear(x,
                 output_size,
                 scope=None,
                 reuse=False,
                 init_w='gaussian',
                 weight_start=0.001,
                 use_bias=True,
                 bias_start=0.0,
                 input_size=None):
  """Performs linear operation. Uses ortho init defined earlier."""
  shape = x.get_shape().as_list()
  with tf.variable_scope(scope or 'linear'):
    if reuse is True:
      tf.get_variable_scope().reuse_variables()

    w_init = None  # uniform
    if input_size is None:
      x_size = shape[1]
    else:
      x_size = input_size
    if init_w == 'zeros':
      w_init = tf.constant_initializer(0.0)
    elif init_w == 'constant':
      w_init = tf.constant_initializer(weight_start)
    elif init_w == 'gaussian':
      w_init = tf.random_normal_initializer(stddev=weight_start)

    w = tf.get_variable(
        'super_linear_w', [x_size, output_size], tf.float32, initializer=w_init)
    if use_bias:
      b = tf.get_variable(
          'super_linear_b', [output_size],
          tf.float32,
          initializer=tf.constant_initializer(bias_start))
      return tf.matmul(x, w) + b
    return tf.matmul(x, w)