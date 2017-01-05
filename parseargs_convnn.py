""" Argument parser for ConvNN """

import argparse as ap


def _args_parse():
	""" Allows for command line hyperparameter inputs to models

	Description
	------------

	--image_size: Sets up the size of a square image, default is 32*32 for CIFAR10
	--features : Sizes of convolutional feature maps, the first one ought to be 3 corresponding to RGB channels
	--norm : Batch-normalisation boolean flag (default is True)
	--ptype : Specifies pooling type:
		- 'max' (default) for max pooling
		- 'average' for average pooling
	--aug : Data augmentation boolean flag (default is False)
	--dropout : Set dropout probability, p = 1.0 (default) implies nothing is dropped out
	--optype : Specifies optimizer type, default set to 'ADAM', if anything else is chosen, RMSProp is used
	--hsize : Specifies the size of the hidden layer in the fully connected layer that follows conv layers
	--bsize : Specifies the batch size for training

	Note
	----
	Assertions and exceptions are not added, please be careful in passing arguments. If in doubt, please allow default values.
	For non-square images appropriate modifications need to be made in helper_nn.py

	"""
	parser = ap.ArgumentParser(description = 'Hyperparameters')
	parser.add_argument('--image_size', type = int, dest = 'img_size', 
	                             action = 'store', default = 32)
	parser.add_argument('--features', type = list, dest = 'conv_features', 
	                             action = 'store', default = [3, 64, 128, 256])
	parser.add_argument('--norm', type = bool, dest = 'normalisation', 
	                               action = 'store', default = True)
	parser.add_argument('--ptype', dest = 'pooling_type', 
	                                  action = 'store', default = 'max')
	parser.add_argument('--aug', type = bool, dest = 'augmentation', 
	                                  action = 'store', default = False)
	parser.add_argument('--dropout', dest = 'dropout_p', 
	                                  action = 'store', default = 1.0)
	parser.add_argument('--optype', type = str, dest = 'optimization', 
	                                  action = 'store', default = 'ADAM')
	parser.add_argument('--hsize', type = int, dest = 'hidden_layer_size', 
	                                  action = 'store', default = 500)
	parser.add_argument('--bsize', type = int, dest = 'batch_size', 
	                                  action = 'store', default = 300)
	parser.add_argument('--eta', type = float, dest = 'learning_rate', 
	                                  action = 'store', default = 0.01)
	flags = parser.parse_args()
	return flags

