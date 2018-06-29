###################################################################
# cnn_WIP.py:
#	CNN to recognize walking for WIP algorithm
#
# github: 
#	https://github.com/sdhanson/cnn-WIP
#
# versions:
#	Python 2.7.5
#	TensorFlow 1.8.x
#
# dependencies:
#	numpy
#	pandas
#
###################################################################
# imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import os
# import os.path as path


import tensorflow as tf
from tensorflow.python.tools import freeze_graph

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

tf.logging.set_verbosity(tf.logging.INFO)

###################################################################
# global variables
# side note: need to initialize variables before creating data graph
# for unity BUT estimator should do that for me 
EPOCHS = 5
BATCH_SIZE = 64
TRAIN_STEPS = 1000

DATA_PATH = './data.csv'
TRAIN_PATH = './train.csv'
TEST_PATH = './test.csv'
# walking is 1, standing is 0
CSV_COLUMN_NAMES = ['Euclidean_Accel', 'Activity']

# Input/output dims are HARDCODED, need to be CHANGED w these vals
WINDOW_SIZE = 3
INPUT_SHAPE = [1, 10, 1] 
LEARNING_RATE = 0.001
###################################################################


###################################################################
# load training and testing data - IDK HOW TO DO THIS YET

def load_data():
	data = pd.read_csv(filepath_or_buffer=DATA_PATH,
						names=CSV_COLUMN_NAMES,
						header=0,
						usecols=[0, 1])

	print (data.head(n=10))
	#debugging

	# data_feature, data_label = data, data.pop(label_name)

	# train_feature, train_label, test_feature, test_label = train_test_split(data_feature, data_label, test_size=0.2)

	# code from before splitting in the map - PROBABLY NEED TO SPLIT
	# THE DATA IN ANOTHER FILE ???? no this is fine bc predict mode won't do this??
	# train = pd.read_csv(filepath_or_buffer=TRAIN_PATH,
	# 					names=CSV_COLUMN_NAMES,
	# 					header=0)
	# train_feature, train_label = train, train.pop(label_name)

	# test = pd.read_csv(filepath_or_buffer=TEST_PATH,
	# 					names=CSV_COLUMN_NAMES,
	# 					header=0)
	# test_feature, test_label = test, test.pop(label_name)

	# (train_feature, train_label), (test_feature, test_label) = load_data()
###################################################################



###################################################################
# define CNN model

def cnn(features, labels, mode):
	# reshapse X to 4-D tensor: [batch_size, width, height, channels]
	input_layer = tf.reshape(features['x'], [-1, 1, 10, 1])

	# Convolutional Layer
	# side note - strides is automatically set to 1
	# input shape: [batch_size, 1, 10, 1]
	# output shape: [batch_size, 1, 10, 128]
	conv = tf.layers.Conv1D(
		inputs=input_layer,
		filters=128,
		kernel_size=WINDOW_SIZE,
		padding='same',
		activation=tf.nn.relu)


	# Pooling Layer
	# pool_size is the size of the kernel ?? strides default to 1 ??
	# input shape: [batch_size, 1, 10, 128]
	# output shape: [batch_size, 1, 3, 128]
	pool = tf.layers.MaxPooling1D(inputs=conv, pool_size=WINDOW_SIZE, padding='same')


	# Flatten Layer
	# input shape: [batch_size, 1, 10, 128]
	# output shape: [batch_size, 1 * 3 * 128]
	pool_flat = tf.reshape(pool, [-1, 1 * 3 * 128])


	# Dense Layer
	# input shape: [batch_size, 1 * 3 * 128]
	# output shape: [batch_size, 364]
	dense = tf.layers.dense(inputs=pool_flat, units=364, activation=tf.nn.relu)

	# Dropout Layer
	dropout = tf.layers.dropout(inputs=dense, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

	# Logits layer
	# input shape: [batch_size, 364]
	# output shape: [batch_size, 2]
	logits = tf.layers.dense(inputs=dropout, units=6)

	predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
		"classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
		"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

	# PREDICT mode: return the CNN predictions
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	# TRAIN and EVAL mode: loss w sigmoid cross entropy given logits
	loss=tf.losses.sigmoid_cross_entropy(labels=labels, logits=logits)

	# TRAIN mode: train with ADAM optimizer
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer= tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
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
# train function
def train_input_fn(features, labels, batch_size):
	dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

	dataset = dataset.shuffle(1000).repeat().batch(batch_size)

	return dataset
###################################################################


###################################################################
# eval function
def eval_input_fn(features, labels=None, batch_size=None):
	if labels is None:
		inputs = features
	else:
		inputs = (features, labels)

	dataset = tf.data.Dataset.from_tensor_slices(inputs)

	assert batch_size is not None, "batch_size must not be None"
	dataset = dataset.batch(batch_size)

	return dataset.make_one_shot_iterator().get_next()
###################################################################



###################################################################
# main
def main():
	# load data
	load_data();
	# (train_feature, train_label), (test_feature, test_label) = load_data()

	# # set feature columns
	# my_feature_columns = []
	# for key in train_feature.keys():
	# 	my_feature_columns.append(tf.feature_column.numeric_column(key=key))

	# # build cnn
	# model_fn = cnn()

	# # create estimator
	# wip_classifier = tf.estimator.Estimator(model_fn=model_fn, 
	# 										params={
	# 											'feature_columns': my_feature_columns,
	# 										})

	# # set up logging
	# tensors_to_log = {'probabilities': 'softmax_tensor'}
	# logging_hook = tf.train.LoggingTensorHook(
	# 	tensors=tensors_to_log, every_n_iter=10)

	# # train cnn
	# train_input_fn = lambda:train_input_fn(train_feature, train_label, BATCH_SIZE)

	# wip_classifier.train(
	# 	input_fn=train_input_fn,
	# 	steps=TRAIN_STEPS,
	# 	hooks=[logging_hook])

	# # freeze cnn - IDK IF THIS IS IN THE RIGHT PLACE
	# freeze_graph.freeze_graph(input_graph = model_path +'/raw_graph_def.pb',
	# 			input_binary = True,
	# 			input_checkpoint = last_checkpoint,
	# 			output_node_names = "action",
	# 			output_graph = model_path +'/frozen_wip_cnn1.bytes' ,
	# 			clear_devices = True, initializer_nodes = "",input_saver = "",
	# 			restore_op_name = "save/restore_all", filename_tensor_name = "save/Const:0")

	# # eval cnn
	# eval_input_fn = lambda:eval_input_fn(train_feature, train_label, BATCH_SIZE)

	# eval_results = wip_classifier.evaluate(input_fn=eval_input_fn)
	# print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

	# set up PREDICTIONS
###################################################################


if __name__ == "__main__":
  main()