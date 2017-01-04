"""Training and testing setup"""

import tensorflow as tf
import model


def loss(logits, labels):
    """Calculates the loss from the logits and the labels.

    Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].

    Returns:
    loss: Loss tensor of type float.
    """
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, labels),
        name='xentropy_mean')
    return loss


def training(loss, FLAGS):
    """Sets up the training Ops.

    Creates a summarizer to track the loss over time in TensorBoard.

    Creates an optimizer and applies the gradients to all trainable variables.

    The Op returned by this function is what must be passed to the
    `sess.run()` call to cause the model to train.

    Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.

    Returns:
    train_op: The Op for training.
    """
    # Add a scalar summary for the snapshot loss.
    tf.scalar_summary(loss.op.name, loss)
    # Create the gradient descent optimizer with the given learning rate.
    if FLAGS.optimization == 'ADAM':
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    else:
        optimizer = tf.train.RMSPropOptimizer(FLAGS.learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op
   


def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label.

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      labels: Labels tensor, int32 - [batch_size], with values in the
        range [0, NUM_CLASSES).

    Returns:
      A scalar int32 tensor with the number of examples (out of batch_size)
      that were predicted correctly.
    """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label is in the top k (here k=1)
    # of all logits for that example.
    labels_int = tf.argmax(labels, 1)
    correct = tf.nn.in_top_k(logits, labels_int, 1)
    # Return the number of true entries.
    return tf.reduce_sum(tf.cast(correct, tf.int32))
