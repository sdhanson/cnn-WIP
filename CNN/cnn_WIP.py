from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import numpy & tensorflow
import numpy as np
import tensorflow as tf

# parameters
window_size = [3, 3]

# network parameters

# set logging threshold
tf.logging.set_verbosity(tf.logging.INFO)

def conv_net(features, labels, mode):

	# input layer
	input_layer = tf.reshape(features["x"], [-1, 1, 10, 1])

	# convolutional layer 
	conv = tf.layers.Conv1D(
		inputs=input_layer,
		filters = 128,
		kernel_size = window_size,
		padding='valid',
		activation=tf.nn.relu)

	# max pooling layer
	pool = tf.layers.MaxPooling1D(
		inputs=conv,
		pool_size=window_size,
		strides=1)

	# dropout layer
	pool_flat = tf.reshape(pool, [-1, SOMEDIMENSIONS])
	dense = tf.layers.dense(
		inputs=pool_flat,
		units=SOMENUMBER,
		activation=tf.nn.relu)
	droupout = tf.layers.dropout(
		inputs=dense,
		rate=0.5,
		training=mode == tf.estimator.ModeKeys.TRAIN)

	# fully-connected layer
	logits = tf.layers.dense(inputs=dropout, units=2)

	# softmax
	predictions = {
		"classes": tf.argmax(input=logits, axis=1),
		"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
	}

	# predictions
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # loss - TRAIN & EVAL
	loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

	# training set-up
 	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
		train_op = optimizer.minimize(
			loss=loss,
			global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # eval
	eval_metric_ops = {
		"accuracy": tf.metrics.accuracy(
			labels=labels, predictions=predictions["classes"])}
		return tf.estimator.EstimatorSpec(
			mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


# Our application logic will be added here

if __name__ == "__main__":
  tf.app.run()