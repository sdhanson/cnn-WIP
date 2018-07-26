# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 18:06:40 2018

@author: adamsha
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras import backend as K

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

import matplotlib.pyplot as plt
from scipy import stats
import os # os.getcwd()
import os.path as path


tf.logging.set_verbosity(tf.logging.INFO)


###################################################

# DATA PREPROCESSING 
# Load the training data into two NumPy arrays. 
def read_data(file_path):
    column_names = ['activity','timestamp', 'x-axis', 'y-axis', 'z-axis']
    data = pd.read_csv(file_path,header = 0, names = column_names)
    return data

# def feature_normalize(dataset):
#     mu = np.mean(dataset,axis = 0)
#     sigma = np.std(dataset,axis = 0)
#     return (dataset - mu)/sigma

def plot_axis(ax, x, y, title):
    ax.plot(x, y)
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)

def plot_activity(activity,data):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows = 3, figsize = (15, 10), sharex = True)
    plot_axis(ax0, data['timestamp'], data['x-axis'], 'x-axis')
    plot_axis(ax1, data['timestamp'], data['y-axis'], 'y-axis')
    plot_axis(ax2, data['timestamp'], data['z-axis'], 'z-axis')
    #plot_axis(ax0, data['timestamp'], data['vector-mag'], 'vector-mag')
    plt.subplots_adjust(hspace=0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()


## SEGMENT SIGNALS
    # ADD DESCRIPTION
    # The windows function will generate indexes as specified by the size 
    # parameter by moving over the signal by fixed step size. The window size 
    # used is 90, which equals to 4.5 seconds of data and as we are moving each 
    # time by 45 points the step size is equal to 2.25 seconds. 
def windows(data, size):
    start = 0
    while start < data.count():
        yield int(start), int(start + size)
        start += (size / 2)
        
def segment_signal(data,window_size = 90):
    segments = np.empty((0,window_size,3))
    labels = np.empty((0))
    for (start, end) in windows(data["timestamp"], window_size):
        x = data["x-axis"][start:end]
        y = data["y-axis"][start:end]
        z = data["z-axis"][start:end]
        if(len(data["timestamp"][start:end]) == window_size):
            segments = np.vstack([segments,np.dstack([x,y,z])])
            labels = np.append(labels,stats.mode(data["activity"][start:end])[0][0])
    return segments, labels


# CONVOLUTIONAL NEURAL NET 
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.0, shape = shape)
    return tf.Variable(initial)
	
def depthwise_conv2d(x, W):
    return tf.nn.depthwise_conv2d(x,W, [1, 1, 1, 1], padding='VALID')
	
def apply_depthwise_conv(x,kernel_size,num_channels,depth):
    weights = weight_variable([1, kernel_size, num_channels, depth])
    biases = bias_variable([depth * num_channels])
    return tf.nn.relu(tf.add(depthwise_conv2d(x, weights),biases, name="input_node"))
    
def apply_max_pool(x,kernel_size,stride_size):
    return tf.nn.max_pool(x, ksize=[1, 1, kernel_size, 1], 
                          strides=[1, 1, stride_size, 1], padding='VALID')

# EXPORT GAPH FOR UNITY
def export_model(saver, input_node_names, output_node_name):
    if not path.exists('out'):
        os.mkdir('out')

    cnn_wip2 = "cnn_wip2_sara";

    tf.train.write_graph(K.get_session().graph_def, 'out', cnn_wip2 + '_graph.pbtxt')

    saver.save(K.get_session(), 'out/' + cnn_wip2 + '.chkp')

    freeze_graph.freeze_graph('out/' + cnn_wip2 + '_graph.pbtxt', None, False,
                              'out/' + cnn_wip2 + '.chkp', output_node_name,
                              "save/restore_all", "save/Const:0",
                              'out/frozen_' + cnn_wip2 + '.bytes', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + cnn_wip2 + '.bytes', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, input_node_names, [output_node_name],
            tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out/opt_' + cnn_wip2 + '.bytes', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("graph saved!")



#window size:
    # 90 = 4.5 seconds
    # 40 = 2 seconds
    # 30 = 1.5 seconds
    # 20 = 1 seconds

# og
# window_size = 90
# kernel_size = 60
# depth = 60
# pooling_filter_size = 20
# kernel_size_2 = 6

# These values work:    
# 90, 60, 60, 20 (6)    
# 40, 20, 20, 10
# 30, 20, 20, 2, (2) [97.08%] ** 
# 20, 10, 10, 2, (2) [91.84%]

    

def main(): 
    # 'Saver' op to save and restore all the variables
    #saver = tf.train.Saver()
    model_path = os.getcwd()  
    model_path = model_path + "/tmp/model.ckpt"

    
    # DATA PREPROCESSING VARS
    visualize = False           # bool - display graph or no
    window_size = 90            # length of sliding window
    input_width = window_size;  # length of input for CNN
    input_height = 1            # 1D data          
        
    num_channels = 3            # num inputs (vm or triaxial)
    num_labels = 2              # num outputs (classification labels)

    # CONVOLUTIONAL NEURAL NET VARS
    # Convolutional Layer 
    batch_size = 10
    kernel_size = 60            # number of channels of output from conv layer
    depth = 60
    num_hidden = 1000
    # Pooling Layer
    pooling_filter_size = 20
    stride = 2
    #Second Confolution
    kernel_size2 = 2            # number of channels of output from conv layer ( was 6 now 2 )
    # Training 
    learning_rate = 0.0001
    training_epochs = 1         #5 is sufficient 
    # total_batches  is set in relation to train_x.shape later
    
    
    print(" \nDATA PREPROCESSING") 
    print("read data") 
    dataset = read_data("./GO_1_raw.csv") # CHANGE ME

    # print("normalize x") 
    # dataset['x-axis'] = feature_normalize(dataset['x-axis'])
    # print("normalize y")
    # dataset['y-axis'] = feature_normalize(dataset['y-axis'])
    # print("normalize z")    
    # dataset['z-axis'] = feature_normalize(dataset['z-axis'])
    
    print("visualize data")    
    if (visualize): 
        for activity in np.unique(dataset["activity"]):
            subset = dataset[dataset["activity"] == activity][:180]
            plot_activity(activity,subset)     
        
        
    print("segment signals")    
    segments, labels = segment_signal(dataset, window_size)
    labels = np.asarray(pd.get_dummies(labels), dtype = np.int8)
    
    reshaped_segments = segments.reshape(len(segments), 1, window_size, num_channels)    

    print("Divide into training and testing set (70/30) randomly")    
    train_test_split = np.random.rand(len(reshaped_segments)) < 0.70 # < 0.70
    train_x = reshaped_segments[train_test_split]
    train_y = labels[train_test_split]
    test_x = reshaped_segments[~train_test_split]
    test_y = labels[~train_test_split]
    
    total_batchs = train_x.shape[0] 

    print(" \n\nSET UP CNN \nDefine Tensorflow placeholders for input/output")    
    X = tf.placeholder(tf.float32, shape=[None,input_height,input_width,num_channels], name="input_placeholder_x")
    Y = tf.placeholder(tf.float32, shape=[None,num_labels])

    # The 1st conv. layer has filter size and depth of 60 
    # (number of channels, we will get as output from convolution layer). 
    print("First convolution layer") 
    c = apply_depthwise_conv(X,kernel_size,num_channels,depth)

    # The pooling layer’s filter size is set to 20 and with a stride of 2. 
    print("Max pooling operation")
    p = apply_max_pool(c,pooling_filter_size,stride) 

    # Next, the conv. layer takes an input of max-pooling layer
    # and applies a filter of size 6. It will have 1/10 of depth of max-pooling layer.
    print("Second convolution layer")     
    c = apply_depthwise_conv(p,kernel_size2,depth*num_channels,depth//10)  # 
    
    # Flatten output for fully connected layer - should have 1000 neurons
    print("Flatten for fully connected layer")     
    shape = c.get_shape().as_list()
    c_flat = tf.reshape(c, [-1, shape[1] * shape[2] * shape[3]])
    
    # Fully connected layer operations
    # Thanh function is used as non-linearity
    print("Tanh operation for non-linearity")    
    f_weights_l1 = weight_variable([shape[1] * shape[2] * depth * num_channels * (depth//10), num_hidden])
    f_biases_l1 = bias_variable([num_hidden])
    f = tf.nn.tanh(tf.add(tf.matmul(c_flat, f_weights_l1),f_biases_l1))
    
    # Softmax layer is defined to output probabilities of the class labels
    print("Softmax outputs probabilities")     
    out_weights = weight_variable([num_hidden, num_labels])
    out_biases = bias_variable([num_labels])
    y_ = tf.nn.softmax(tf.matmul(f, out_weights) + out_biases, name="output_node")    
    
    
    # Minimise negtive log-likelihood cost function using stochastic gradient descent optimiszer
    # Initialize cost function and optimizer. 
    print("Gradient descent optimizer")
    loss = -tf.reduce_sum(Y * tf.log(y_))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)

    #Define code for accuracy calculuation of the prediction by model    
    print("Predictions")
    correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    
    print(" \nTRAIN, TEST, AND PRAY") 
    
    with tf.Session() as session:
        tf.global_variables_initializer().run()
        for epoch in range(training_epochs):
            #cost_history = np.empty(shape=[1],dtype=float)
            cost_history = np.empty(shape=[0],dtype=float)            
            for b in range(total_batchs):    
                offset = (b * batch_size) % (train_y.shape[0] - batch_size)
                batch_x = train_x[offset:(offset + batch_size), :, :, :]
                batch_y = train_y[offset:(offset + batch_size), :]
                _, c = session.run([optimizer, loss],feed_dict={X: batch_x, Y : batch_y})
                cost_history = np.append(cost_history,c)
            print("Epoch: ",epoch," Training Loss: ",np.mean(cost_history)," Training Accuracy: ",
                  session.run(accuracy, feed_dict={X: train_x, Y: train_y}))
    
        print("Testing Accuracy:", session.run(accuracy, feed_dict={X: test_x, Y: test_y}))

                # Save model weights to disk
        saver = tf.train.Saver()
        save_path = saver.save(session, model_path)
        print("Model saved in file: %s" % save_path)
        # note: you would use
        # saver.restore(sess, model_path)
        # ro restore model weights from prv. saved model
    
        export_model(tf.train.Saver(), ["input_node"], "output_node")
    
    
    
    
    print("finish")
    

# Run program 
main()