""" Helper functions for neural networks """

import tensorflow as tf
import math

POOLING_SIZE = 2
KERNEL_SIZE = 5

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x, size):
    return tf.nn.max_pool(x, ksize=[1, size, size, 1],
                          strides=[1, size, size, 1], padding='SAME')


def avg_pool_2x2(x, size):
    return tf.nn.avg_pool(x, ksize=[1, size, size, 1],
                          strides=[1, size, size, 1], padding='SAME')


def convOp(nprev, nfinal, x, insize, norm=True, ptype='max', first='norm'):
    """Gives the convolutional layer output
    nprev   : number of features from last layer
    nfinal  : number of features from this layer
    x       : input to the layer

    Optional Arguments:
    norm    : set true if normalisation needed
    ptype   : Max or avg pooling
    first   : Sets order of pooling and normalisation
    """
    W = weight_variable([KERNEL_SIZE, KERNEL_SIZE, nprev, nfinal])
    b = bias_variable([nfinal])
    h = tf.nn.relu(conv2d(x, W) + b)
    pooler = max_pool_2x2 if ptype == 'max' else avg_pool_2x2
    if first == 'norm':
        h_norm = tf.nn.local_response_normalization(h)
        h_pool = pooler(h_norm, POOLING_SIZE)
        hf = h_pool
    else:
        h_pool = pooler(h, POOLING_SIZE)
        h_norm = tf.nn.local_response_normalization(h_pool)
        hf = h_norm
    return hf, int(math.ceil(insize//POOLING_SIZE))


def convlayers(x, FLAGS):
	"""sets up the convolution layers of the network"""
	features = FLAGS.conv_features
	num_conv_layers = len(features) - 1 # first element in features is input
	inp = x
	size = FLAGS.img_size
	for i in xrange(num_conv_layers):
		out, size = convOp(features[i], features[i+1], inp, size,
		                   norm=FLAGS.normalisation,
		                   ptype=FLAGS.pooling_type)
		inp = out
	return out, size, features[-1]


def fcOp(nprev, nfinal, x, keep_prob = 1.0, final = False):
	W = weight_variable([nprev, nfinal])
	b = bias_variable([nfinal])
	h = tf.nn.relu(tf.matmul(x, W) + b)
	if not final:
		h_out = tf.nn.dropout(h, keep_prob)
	else:
		h_out = h
	return h_out

def fclayers(x, x_size, layer_sizes, FLAGS):
	n = len(layer_sizes) - 1
	inp = x
	size = x_size
	for i in range(n):
		out = fcOp(size, layer_sizes[i], inp, 
					keep_prob=FLAGS.dropout_p,
					final = False)
		inp = out
		size = layer_sizes[i]

	out = fcOp(size, layer_sizes[-1], inp, final = True)
	return out

