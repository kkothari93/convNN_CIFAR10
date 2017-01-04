import helper_nn as hnn
import tensorflow as tf

NUM_CLASSES = 10

def network(x, FLAGS):
	""" Sets up the neural network
	
	Builds in two parts: Convolution layers interspersed with pooling layers 
	followed by fully connected layers
	"""
	conv_out, outsize, nfeatures = hnn.convlayers(x, FLAGS)
	conv_out_flat = tf.reshape(conv_out, [-1, outsize * outsize * nfeatures])
	y = hnn.fclayers(conv_out_flat, outsize * outsize * nfeatures, [FLAGS.hidden_layer_size, NUM_CLASSES], FLAGS)
	return y