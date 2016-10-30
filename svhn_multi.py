#! /usr/local/bin/python

# Lot of the code here is from tensorflow examples

import sys
import os
import math
import cPickle as pickle
import numpy as np
import operator
import tensorflow as tf
from scipy.misc import imsave

# ********************************************************************************
# Load and format the data
# ********************************************************************************
pickle_file = 'SVHN_multi_48.pickle'
with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

# ********************************************************************************
# Model parameters
# ********************************************************************************

IMAGE_SIZE = 48
NUM_LABELS = 11 # digits 0-9 and additional label to indicate absence of a digit(10)
BATCH_SIZE = 64
LEARNING_RATE = 0.00005
LAMBDA = 0.0005 # regularization rate
NUM_STEPS = 200000
NUM_CHANNELS = 3
NUM_DIGITS = 5 # number of letters in the sequence to transcribe
RESTORE = True # set to False to run the training from scratch
MODEL_CKPT = 'model.ckpt' # checkpoint file
CDEPTH1 = 16
CDEPTH2 = 32
CDEPTH3 = 64
LOG_DIR = 'logs.{}'.format(os.getpid()) # where to write summary logs
N_HIDDEN_1 = (IMAGE_SIZE // 4) * (IMAGE_SIZE // 4) * CDEPTH3 # 4 => 2^2 (2 = number of pooling layers)

def reformat(dataset, labels):
    labels = labels[:, 0:NUM_DIGITS+1]
    return dataset, labels

print("After reformatting ... ")
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

# *** SEEME ***:
# use a small set for validation and test for now
# as the system needs tons of RAM to do convolutions
# on a larger set. We need faster turnaround for now.
valid_dataset = valid_dataset[:200, :]
valid_labels = valid_labels[:200]
test_dataset = test_dataset[:1000, :]
test_labels = test_labels[:1000]

# *** SEEME ***:
# used for validating an architecture
# on a very small dataset, if the model overfits to 100% minibatch or training accuracy,
# model is about right and hyperparameter tuning is required.
validate_arch = False
if validate_arch:
    print("Validating architecture")
    train_dataset = train_dataset[:100, :]
    train_labels = train_labels[:100]
    valid_dataset = valid_dataset[:10, :]
    valid_labels = valid_labels[:10]
    test_dataset = test_dataset[:10, :]
    test_labels = test_labels[:10]
    BATCH_SIZE = 10
    NUM_STEPS = 5000
    LAMBDA = 0.0
    MODEL_CKPT = 'model_valid.ckpt'
    LOG_DIR = 'valid_logs'
    RESTORE = False # never restore for validation

print('Inputs to the model')
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

# ********************************************************************************
# Various helper functions
# ********************************************************************************

# add summary nodes for the variable
def variable_summaries(var, name):

  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.scalar_summary('mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.scalar_summary('stddev/' + name, stddev)
    tf.scalar_summary('max/' + name, tf.reduce_max(var))
    tf.scalar_summary('min/' + name, tf.reduce_min(var))
    tf.histogram_summary(name, var)

# weight variable for the network
def weight_variable(name, shape):
  # name: name of this variable
  # shape: list of shape compatible with tf.Variable call
  fan_in = shape[-2]
  fan_out = shape[-1]
  for x in shape[:-2]:
    fan_in *= x
    fan_out *= x

  stddev = math.sqrt(2.0/fan_in)
  var = tf.Variable(tf.truncated_normal(shape, 0.0, stddev=stddev, name=name))
  # add variable to the summaries for visualization
  variable_summaries(var, name)
  return var

# bias variable for the network
def bias_variable(name, shape):
  # name: name of the variable
  # shape: list representing shape of Tensor. compatible with tf shape
  var = tf.constant(0.01, shape=shape)
  var = tf.Variable(var)
  variable_summaries(var, name)
  return var

# convert a list of logits to probabilities by applying softmax on them
def logitss_to_probs(logitss):
    # input: a list of logits
    # output: a 2-D array of softmax operations (they have to be eval'ed in tf session)
    # just applies softmax on each of the logits
    return map(tf.nn.softmax, logitss)

# estimate the accuracy of the predictions
def tf_accuracy(predictions, tf_labels):
  # predictions is a list of logits for each classifier.

  # add an argmax op for each classifier
  xs = [tf.argmax(p, 1) for p in predictions]

  # pack the results
  pred_labels = tf.pack(xs, axis=1)

  # convert 2-D array of booleans to vector of bools
  # we say that an example is correctly classified only when all labels are correct
  results = tf.reduce_all(tf.equal(pred_labels, tf_labels), 1)

  # accuracy is the number of correct predictions to total number of predictions
  accuracy = 100 * tf.reduce_mean(tf.cast(results, tf.float32))

  return accuracy

# ********************************************************************************
# Setup the variables
# ********************************************************************************

# Input data. For the training data, we use a placeholder that will be fed
# at run time with a training minibatch.
with tf.name_scope('inputs'):
  tf_train_dataset = tf.placeholder(tf.float32,
                                  shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
  tf.image_summary('input', tf_train_dataset, max_images=10)
  # 6 here is 1 digit for length of sequence and 5 for digits themselves
  tf_train_labels = tf.placeholder(tf.int64, shape=(BATCH_SIZE, NUM_DIGITS+1))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf.image_summary('validation', tf_valid_dataset, max_images=10)
  tf_test_dataset = tf.constant(test_dataset)
  tf.image_summary('test', tf_test_dataset, max_images=10)

# Store layers weight & bias
# after 2 max pooling operations, the feature maps will have 1/(2*2) of the original spatial dimensions
weights = {
  'conv1': weight_variable('conv1/weights', [5, 5, NUM_CHANNELS, CDEPTH1]), # 5x5 kernel, depth CDEPTH1
  'conv2': weight_variable('conv2/weights', [5, 5, CDEPTH1, CDEPTH2]), # 5x5 kernel, depth CDEPTH2
  'conv3': weight_variable('conv3/weights', [5, 5, CDEPTH2, CDEPTH3]), # 5x5 kernel, depth CDEPTH3
  # for the length of the sequence of digits
  'out1': weight_variable('out1/weights', [N_HIDDEN_1, 5]), # length of the sequence: here 1-5 - TODO: make it configurable
  }

# for individual digits
for i in range(2, NUM_DIGITS+2):
  weights['out{}'.format(i)] = weight_variable('out{}/weights'.format(i), [N_HIDDEN_1, NUM_LABELS])

biases = {
  'conv1': bias_variable('conv1/bias', [CDEPTH1]),
  'conv2': bias_variable('conv2/bias', [CDEPTH2]),
  'conv3': bias_variable('conv3/bias', [CDEPTH3]),
  # for the length of sequence: here 1-5
  'out1': bias_variable('out1/bias', [5]),
  }

# for individual digits
for i in range(2, NUM_DIGITS+2):
  biases['out{}'.format(i)] = bias_variable('out{}/bias'.format(i), [NUM_LABELS])

# ********************************************************************************
# Helper to setup the convolution net
# ********************************************************************************

def setup_conv_net(X, weights, biases, train=False):

  # convolution layers with ReLU activations and max pooling
  conv = tf.nn.conv2d(X,
                      weights['conv1'],
                      strides=[1, 1, 1, 1],
                      padding='SAME', name='conv1')
  relu = tf.nn.relu(tf.nn.bias_add(conv, biases['conv1']), name='relu1')
  if train:
    relu = tf.nn.dropout(relu, 0.8)
  pool = tf.nn.max_pool(relu, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
  print("Pool1 shape: " + str(pool.get_shape().as_list()))

  conv = tf.nn.conv2d(pool,
                      weights['conv2'],
                      strides=[1, 1, 1, 1],
                      padding='SAME', name='conv2')
  relu = tf.nn.relu(tf.nn.bias_add(conv, biases['conv2']), name='relu2')
  if train:
    relu = tf.nn.dropout(relu, 0.6)
  pool = tf.nn.max_pool(relu, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name='pool2')
  print("Pool2 shape: " + str(pool.get_shape().as_list()))

  conv = tf.nn.conv2d(pool,
                      weights['conv3'],
                      strides=[1, 1, 1, 1],
                      padding='SAME', name='conv3')
  relu = tf.nn.relu(tf.nn.bias_add(conv, biases['conv3']), name='relu3')
  if train:
    relu = tf.nn.dropout(relu, 0.5)
  print("Pool3 shape: " + str(relu.get_shape().as_list()))

  # reshape the resulting cuboid to feed to the
  # downstream fully connected layers
  shape = relu.get_shape().as_list()
  reshape = tf.reshape(relu,
                       [shape[0], shape[1] * shape[2] * shape[3]])

  logitss = []
  logits = tf.nn.bias_add(tf.matmul(reshape, weights['out1']), biases['out1'])
  logitss.append(logits)

  for i in range(2, NUM_DIGITS+2):
    out = 'out{}'.format(i)
    logits = tf.nn.bias_add(tf.matmul(reshape, weights[out]), biases[out])
    logitss.append(logits)

  return logitss

logitss = setup_conv_net(tf_train_dataset, weights, biases, train=True)

# ********************************************************************************
# Setup the training loss operations
# ********************************************************************************

loss =  tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logitss[0], tf_train_labels[:, 0]))

# regularization loss of required
if LAMBDA > 0.0:
  loss += LAMBDA * tf.nn.l2_loss(weights['conv1'])
  loss += LAMBDA * tf.nn.l2_loss(weights['conv2'])
  loss += LAMBDA * tf.nn.l2_loss(weights['conv3'])
  loss += LAMBDA * tf.nn.l2_loss(biases['conv1'])
  loss += LAMBDA * tf.nn.l2_loss(biases['conv2'])
  loss += LAMBDA * tf.nn.l2_loss(biases['conv3'])
  loss += LAMBDA * tf.nn.l2_loss(weights['out1'])
  loss += LAMBDA * tf.nn.l2_loss(biases['out1'])

  for i in range(2, NUM_DIGITS+2):
    loss += LAMBDA * tf.nn.l2_loss(weights['out{}'.format(i)])
    loss += LAMBDA * tf.nn.l2_loss(biases['out{}'.format(i)])

# add a summary for loss
tf.scalar_summary('training loss', loss)

# ********************************************************************************
# Setup the optimizer
# ********************************************************************************

optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

# ********************************************************************************
# Predictions for the training, validation data
# ********************************************************************************

train_prediction = logitss_to_probs(logitss)
train_accuracy = tf_accuracy(train_prediction, tf_train_labels)
tf.scalar_summary('training accuracy', train_accuracy)

valid_logitss = setup_conv_net(tf_valid_dataset, weights, biases)
valid_prediction = logitss_to_probs(valid_logitss)
tf_valid_labels = tf.constant(valid_labels, dtype=tf.int64)
valid_accuracy = tf_accuracy(valid_prediction, tf_valid_labels)
tf.scalar_summary('validation accuracy', valid_accuracy)

# Test data predictions
test_logitss = setup_conv_net(tf_test_dataset, weights, biases)
test_prediction = logitss_to_probs(test_logitss)
tf_test_labels = tf.constant(test_labels, dtype=tf.int64)
test_accuracy = tf_accuracy(test_prediction, tf_test_labels)

# ********************************************************************************
# Setup validation loss
# ********************************************************************************

vloss =  tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(valid_logitss[0], tf_valid_labels[:, 0]))
for i in range(2, NUM_DIGITS+2):
  vloss += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(valid_logitss[i-1], tf_valid_labels[:, i-1]))

tf.scalar_summary('validation loss', vloss)

# ********************************************************************************
# Setup the session, merge all the summaries and write them out to ./logs
# ********************************************************************************

session = tf.Session()
merged = tf.merge_all_summaries()
train_writer = tf.train.SummaryWriter(LOG_DIR + '/train',
                                      session.graph)

init = tf.initialize_all_variables()
saver = tf.train.Saver()

# ********************************************************************************
# Run the graph in the session now
# ********************************************************************************

with session.as_default():
  # Start running the graph operatons
  if not RESTORE:
    session.run(init)
    print("Initialized")
  else:
    saver.restore(session, MODEL_CKPT)
    print("Restored")

  if not RESTORE:

    # run the training steps if we didn't restore a stored model
    for step in range(NUM_STEPS):

      # Pick an offset within the training data, which has been randomized.
      # Note: we could use better randomization across epochs.
      offset = (step * BATCH_SIZE) % (train_labels.shape[0] - BATCH_SIZE)
      # Generate a minibatch.
      batch_data = train_dataset[offset:(offset + BATCH_SIZE), :]
      batch_labels = train_labels[offset:(offset + BATCH_SIZE), 0:NUM_DIGITS+1]
      # Prepare a dictionary telling the session where to feed the minibatch.
      # The key of the dictionary is the placeholder node of the graph to be fed,
      # and the value is the numpy array to feed to it.
      feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
      _, l, predictions = session.run(
        [optimizer, loss, train_prediction], feed_dict=feed_dict)

      if (step % 10 == 0): # important to do this in closer steps to get a better feel of the loss value
        print("Minibatch loss at step %d: %f" % (step, l))
        train_acc = session.run(train_accuracy, feed_dict=feed_dict)
        print("Minibatch accuracy: %.1f%%" % train_acc)
        val_acc = session.run(valid_accuracy)
        print("Validation accuracy: %.1f%%" % val_acc)

        summary = session.run(merged, feed_dict=feed_dict)
        train_writer.add_summary(summary, step)

    # store the model for restoration later
    saved_in = saver.save(session, MODEL_CKPT)
    print("Model saved in {}".format(saved_in))

  # predict the test labels
  test_acc = session.run(test_accuracy)
  print("Test accuracy: %.1f%%" % test_acc)

train_writer.close()
