###################################################################
# cnn_WIP.py:
#	CNN to recognize walking for WIP algorithm
#
# github: 
#
#
# versions:
#	Python 2.7.5
#	TensorFlow 1.8.x
#	Keras
#
# dependencies:
#	os
#	matplotlib
#	numpy
#
###################################################################
# imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as path

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

###################################################################
# global variables
epochs = 200
batch_size = 64

# Input/output dims are HARDCODED, need to be CHANGED w these vals
window_size = 3
input_shape = [1, 10, 1] 
learning_rate = 

###################################################################


###################################################################
# network parameters

###################################################################


###################################################################
# load training and testing data

def load():
	d_train
	d_test


###################################################################


###################################################################
# define CNN model - SWITCH TO TENSOR FLOW TO COPY THEIR STUFF

def cnn():
	# reshapse X to 4-D tensor: [batch_size, width, height, channels]
	input_layer = tf.reshape(features['x'], [batch_size, 1, 10, 1])

	# Convolutional Layer
	# side note - strides is automatically set to 1
	# input shape: [batch_size, 1, 10, 1]
	# output shape: [batch_size, 1, 10, 128]
	conv = tf.layers.Conv1D(
		inputs=input_layer,
		filters=128,
		kernel_size=window_size,
		padding='same',
		activation=tf.nn.relu)


	# Pooling Layer
	# pool_size is the size of the kernel ?? strides default to 1 ??
	# input shape: [batch_size, 1, 10, 128]
	# output shape: [batch_size, 1, 3, 128]
	pool = tf.layers.MaxPooling1D(inputs=conv, pool_size=window_size, padding='same')


	# Flatten Layer
	# input shape: [batch_size, 1, 10, 128]
	# output shape: [batch_size, 1 * 3 * 128]
	pool_flat = tf.reshape(pool, [batch_size, 1 * 3 * 128])


	# Dense Layer
	# input shape: [batch_size, 1 * 3 * 128]
	# output shape: [batch_size, 364]
	dense = tf.layers.dense(inputs=pool_flat, units=364, activation=tf.nn.relu)

	# Dropout Layer
	dropout = tf.layers.dropout(inputs=dense, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

	# Logits layer
	# input shape: [batch_size, 364]
	# output shape: [batch_size, 2]
	logits = tf.layers.dense(inputs=dropout, units=2)

	predictions = {
		# finds the index of the largest value in the tensor
		# IDK WHY THIS IS HELPFUL - later used in EVAL mode ??
		'classes': tf.argmax(input=logits, axis=1)

		# use softmax to calculate the result
		'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
	}

	# PREDICT mode: return the CNN predictions
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	# TRAIN and EVAL mode: loss w sigmoid cross entropy given logits
	loss=tf.losses.sigmoid_cross_entropy(labels=labels, logits=logits)

	# TRAIN mode: train with ADAM optimizer
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer= tf.train.AdamOptimizer(learning_rate=learning_rate)
		train_op = optimizer.minimize(
			loss=loss,
			global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

	# EVAL mode: add metrics
	eval_metric_ops = {
		'accuracy': tf.metrics.accuracy(
			labels=labels, predictions=predictions['classes'])}
	return tf.estimator.EstimatorSpec(
		mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

###################################################################


###################################################################
# train the CNN

def train(model, d_train, d_test):
	model.compile(loss=WANT SIGMOID BLAH BLAH, optimiser=keras.optimizers.ADAPTIVEMOVEMENTESTIMATION()),
					metrics=['accuracy'])
	model.fit(d_train, batch_size=batch_size, epochs=epochs, verbose=True, validation_data=(d_test))

###################################################################


###################################################################
# freeze CNN for Unity
def freeze(saver, model, input_node_names, output_node_name):

###################################################################

###################################################################
# main
def main():
	# load data

	# create estimator
	wip_classifier = tf.estimator.Estimator(
		model_fn=cnn, model_dir='tmp/wip_cnn_model')

	# set up logging
	tensors_to_log = {'probabilities': 'softmax_tensor'}
	logging_hook = tf.train.LoggingTensorHook(
		tensors=tensors_to_log, every_n_iter=10)

	# build cnn
	model = cnn()

	# train cnn - LOOK INTO THIS AND THE SHUFFLE = TRUE THING, x, y, and steps
	train_input_fn = tf.estimator.inputs.numpy_input_fn(
		x=IDK,
		y=IDK,
		batch_size=batch_size,
		num_epochs=epochs,
		shuffle=True)

	wip_classifier.train(
		input_fn=train_input_fn,
		steps=IDK,
		hooks=[logging_hook])

	eval_input_fn = tf.estimator.inputs.numpy_input_fn(
		x=IDK,
		y=IDK,
		num_epochs=1,
		shuffle=False)
	eval_results = wip_classifier.evaluate(input_fn=eval_input_fn)
	print(eval_results)

	# freeze cnn - LOOK UP HOW TO DO THIS THIS CODE IS LEFT OVER
	freeze_model(tf.train.Saver(), model, ["conv1d_1_input"], "dense_1/Softmax")

###################################################################


if __name__ == "__main__":
  main()