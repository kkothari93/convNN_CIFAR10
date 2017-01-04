""" Trains and evaluates the ConvNN using a feed dictionary. """

import helper_data as hd
import model
import train
import parseargs_convnn
import tensorflow as tf
import time

def placeholder_inputs(batch_size, image_size):
    """Generate placeholder variables to represent the input tensors.

    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded data in the .run() loop, below.

    Args:
      batch_size: The batch size will be baked into both placeholders.

    Returns:
      images_placeholder: Images placeholder.
      labels_placeholder: Labels placeholder.
    """
    image_shape = (image_size, image_size, 3)
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, ) +
                                        image_shape)
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size,
                                                         model.NUM_CLASSES))
    return images_placeholder, labels_placeholder

def run_training():
	""" Train CIFAR10 for a number of steps. """
	FLAGS = parseargs_convnn._args_parse()
	learning_rate = 0.1
	data = hd.CIFAR10Batcher(FLAGS)
	
	with tf.Graph().as_default():
		# Generate placeholders for the images and labels
		images, labels = placeholder_inputs(FLAGS.batch_size, FLAGS.img_size)

		# Get the network output
		logits = model.network(images, FLAGS)

		# Get loss and add loss summary to tensorboard
		loss = train.loss(logits, labels)

		# Add the training op
		train_op = train.training(loss, FLAGS)

		# Add the evaluation Op
		eval_correct = train.evaluation(logits, labels)

		# Build a summary Tensor
		summary = tf.merge_all_summaries()

		# create a saver for saving checkpoints
		saver = tf.train.Saver()

		# Create a session for running the Ops in the Graph
		sess = tf.Session()

		# Instantitate a SummaryWriter to output summaries 
		summary_writer = tf.train.SummaryWriter('./ckpts', sess.graph)

		# Now, run the training

		# Initialise the variables
		sess.run(tf.initialize_all_variables())

		# Start the training loop
		for step in xrange(30000):
			images_batch, labels_batch = data.next_batch(train = True)
			start_time = time.time()

			feed_dict = { images : images_batch, labels : labels_batch}

			_, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

			duration = time.time() - start_time

			# Write the summaries
			if step % 100 == 0:
				# Print to console
				print('Step %d: Loss = %.2f (%.3f sec)' % (step, loss_value, duration))
				# Update the events file
				summary_str = sess.run(summary, feed_dict = feed_dict)
				summary_writer.add_summary(summary_str, step)
				summary_writer.flush()

			# Save a checkpoint
			if (step + 1) % 1000 == 0:
				saver.save(sess, 'checkpoint', global_step=step)
				# Evaluate against the training set
				correct_predictions = sess.run([eval_correct], feed_dict = feed_dict)
				# Calculate precision
				precision = correct_predictions//FLAGS.batch_size
				print('Training Data Eval:')
				print('Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
	        		(FLAGS.batch_size, correct_predictions, precision))

		# Check final test accuracy over n_eval random batches of data in test set
		correct_predictions = 0
		n_eval = 10
		for step in xrange(n_eval):
			images_batch, labels_batch = data.next_batch(train = False)
			feed_dict = { images : images_batch, labels : labels_batch}
			correct_predictions += sess.run([eval_correct], feed_dict = feed_dict)
		
		# Calculate precision
		precision = correct_predictions//(FLAGS.batch_size*n_eval)
		print('Testing Data Eval:')
		print('Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
    		(FLAGS.batch_size*n_eval, correct_predictions, precision))


def main(_):
	run_training()

if __name__ == '__main__':
	tf.app.run()