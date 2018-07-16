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
    column_names = ['activity','timestamp', 'vector-mag']
    data = pd.read_csv(file_path,header = 0, names = column_names)
    return data

def feature_normalize(dataset):
    mu = np.mean(dataset,axis = 0)
    sigma = np.std(dataset,axis = 0)
    return (dataset - mu)/sigma

def plot_axis(ax, x, y, title):
    ax.plot(x, y)
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) + np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)

def plot_activity(activity,data):
    fig, (ax0) = plt.subplots(nrows = 1, figsize = (15, 10), sharex = True)
    #plot_axis(ax0, data['timestamp'], data['x-axis'], 'x-axis')
    #plot_axis(ax1, data['timestamp'], data['y-axis'], 'y-axis')
    #plot_axis(ax2, data['timestamp'], data['z-axis'], 'z-axis')
    plot_axis(ax0, data['timestamp'], data['vector-mag'], 'vector-mag')
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
        
        
def segment_signal(data,window_size):
    segments = np.empty((0,window_size,1))
    labels = np.empty((0)) 
    
    for (start, end) in windows(data["timestamp"], window_size):
        x = data["vector-mag"][start:end] 
        if(len(data["timestamp"][start:end]) == window_size):
            segments = np.vstack([segments,np.dstack([x])])
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
    model_path = os.getcwd()  
    model_path = model_path + "/tmp/model.ckpt"

    
    # DATA PREPROCESSING VARS
    visualize = False           # bool - display graph or no
    window_size =90            # length of sliding window
    input_width = window_size;  # length of input for CNN
    input_height = 1            # 1D data          
        
    num_channels = 1            # num inputs (vm or triaxial)
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
    kernel_size2 = 2            # number of channels of output from conv layer
    # Training 
    learning_rate = 0.0001
    training_epochs = 1         #5 is sufficient 
    # total_batches  is set in relation to train_x.shape later
    
    
    print(" \nDATA PREPROCESSING") 
    print("read data") 
    # dataset = read_data("../WISDM/WISDM_haley_label.csv")
    dataset = read_data("./GO_1_raw.csv")

    print("normalize feature")
    dataset['vector-mag'] = feature_normalize(dataset['vector-mag'])
    
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

    # predict = np.array([12.709550981655466, 12.365693499871487, 11.9365623066519, 18.75197299019742, 14.13980370238212, 7.105775849337265, 12.02733033234444, 15.638324311562357, 15.090546522726406, 2.348947395702892, 21.3946709028614, 6.664262168936164, 3.538381673949756, 12.290882773515195, 20.763759173927177, 7.5378257300658955, 8.579457784571753, 16.913099543751407, 11.852567203805565, 14.459947574018042, 14.216335744744201, 5.389150176038576, 2.856151616994586, 20.933269449568478, 5.740697695241398, 17.341403949056033, 17.71585285829032, 12.003514442589022, 2.983446894455462, 20.64843851447559, 5.459636204889761, 9.045736429839657, 13.689053026421204, 16.683347103156624, 3.2137672049969828, 19.014414825245456, 6.976557237195206, 3.4406950958892057, 14.970546728950971, 20.589399041677986, 6.835474332907062, 1.1032482033333335, 21.746053069233387, 3.5248553755666774, 17.423342833096026, 20.692280235991102, 15.461890767380327, 2.86004593582847, 21.194968234322, 7.695616909785536, 6.248058936380108, 14.29099757160867, 15.552051314530296, 4.958479963446904, 2.492970327698057, 20.764143304285074, 6.981209175211687, 11.454865866388165, 16.017580478433402, 6.158942824170639, 3.7567229510984532, 20.360798990603225, 10.09811705660116, 17.099884443872615, 17.684594758442667, 11.782671530713117, 1.664749702167296, 20.94355148740615, 3.9844580800194813, 9.296643125948501, 16.427437598789172, 15.671274632372299, 2.4320999075455454, 10.068385428322971, 11.443637046125021, 10.544383670597133, 20.053323632260813, 6.7193607042518515, 12.897771426384601, 18.85089537346761, 11.07368597014265, 11.545532982030812, 12.629183740194362, 4.457538450474296, 2.8478580709497057, 21.19272732124258, 5.633327137155788, 17.927058197916754, 15.821579808533397, 16.070312906650404])
    # predict_x = predict.reshape(1, 1, window_size, num_channels)
    
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
        # print("Prediction:", session.run(y_, feed_dict={X: predict_x}))
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