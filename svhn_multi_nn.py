#! /usr/local/bin/python

# Lot of the code here is from tensorflow examples

import sys
import math
import cPickle as pickle
import numpy as np
import operator
import tensorflow as tf

# load and format the data
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

# parameters
IMAGE_SIZE = 48
NUM_LABELS = 11 # digits 0-9 and additional label to indicate absence of a digit(10)
BATCH_SIZE = 128
N_HIDDEN_1 = 64 
LEARNING_RATE = 0.001
LAMBDA = 0.00001 # regularization rate
NUM_STEPS = 5000
NUM_CHANNELS = 1
# number of letters in the sequence to transcribe
NUM_LETTERS = 3
STDDEV = 'fanIn' 
RESTORE = False
MODEL_CKPT = 'model_nn.ckpt'

def reformat(dataset, labels):
    #dataset = dataset.reshape(-1, 3, IMAGE_SIZE, IMAGE_SIZE)
    dataset = dataset.mean(axis=3) # convert to grayscale
    dataset = dataset.reshape((-1, IMAGE_SIZE * IMAGE_SIZE)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    # labels = (np.arange(1,11) == labels[:,None]).astype(np.float32)
    return dataset, labels

print("After reformatting ... ")
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

# *** SEEME ***:
# used for validating an architecture
# on a small dataset, if the model overfits to 100% minibatch or training accuracy,
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

print('Inputs to the model')
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

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

def weight_variable(name, shape, stddev=1.0):
  # name: name of this variable
  # shape: list of shape compatible with tf.Variable call
  # stddev: 'fanIn' - variables should have std_dev 2/sqrt(fan_in)
  #         any float - use verbatim
  if stddev == 'fanIn':
    stddev = math.sqrt(2.0/shape[0])

  var = tf.Variable(tf.truncated_normal(shape, stddev=stddev, name=name))
  # add variable to the summaries for visualization
  variable_summaries(var, name)
  return var

def bias_variable(name, shape):
  # name: name of the variable
  # shape: list representing shape of Tensor. compatible with tf shape
  var = tf.constant(0.1, shape=shape)
  var = tf.Variable(var)
  variable_summaries(var, name)
  return var

def probs_to_labels(probs):
    # input: 2-D array of probabilities (result of softmax)
    # output: a list of labels of size probs.shape[0]
   return [np.argmax(x) for x in probs]

def logitss_to_probs(logitss):
    # input: a list of logits
    # output: a 2-D array of softmax operations (they have to be eval'ed in tf session)
    # just applies softmax on each of the logits
    return map(tf.nn.softmax, logitss)

# Input data. For the training data, we use a placeholder that will be fed
# at run time with a training minibatch.
with tf.name_scope('inputs'):
  tf_train_dataset = tf.placeholder(tf.float32,
                                  shape=(BATCH_SIZE, IMAGE_SIZE * IMAGE_SIZE))
  # 6 here is 1 digit for length of sequence and 5 for digits themselves
  tf_train_labels = tf.placeholder(tf.int32, shape=(BATCH_SIZE, 6))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)

# Store layers weight & bias
# after 2 max pooling operations, the feature maps will have 1/(2*2) of the original spatial dimensions
weights = {
  # for the length of the sequence of digits
  'fc1': weight_variable('fc1/weights', [IMAGE_SIZE * IMAGE_SIZE, N_HIDDEN_1], stddev=STDDEV),
  'out1': weight_variable('out1/weights', [N_HIDDEN_1, 5], stddev=STDDEV), # length of the sequence: here 1-5 - TODO: make it configurable
  }

# for individual digits
for i in range(2, NUM_LETTERS+2):
  weights['fc{}'.format(i)] = weight_variable('fc{}/weights'.format(i), [IMAGE_SIZE * IMAGE_SIZE, N_HIDDEN_1], stddev=STDDEV)
  weights['out{}'.format(i)] = weight_variable('out{}/weights'.format(i), [N_HIDDEN_1, NUM_LABELS], stddev=STDDEV)

biases = {
  # for the length of sequence: here 1-5
  'fc1': bias_variable('fc1/bias', [N_HIDDEN_1]),
  'out1': bias_variable('out1/bias', [5]),
  }

# for individual digits
for i in range(2, NUM_LETTERS+2):
  biases['fc{}'.format(i)] = bias_variable('fc{}/bias'.format(i), [N_HIDDEN_1])
  biases['out{}'.format(i)] = bias_variable('out{}/bias'.format(i), [NUM_LABELS])

def setup_nn(X, weights, biases, train=False):

  logitss = []
  hidden = tf.nn.relu(tf.matmul(X, weights['fc1']) + biases['fc1'], name='fc1')
  logits = tf.matmul(hidden, weights['out1']) + biases['out1']
  logitss.append(logits)

  for i in range(2, NUM_LETTERS+2):
    fc = 'fc{}'.format(i)
    out = 'out{}'.format(i)
    hidden = tf.nn.relu(tf.matmul(X, weights[fc]) + biases[fc], name=fc)
    logits = tf.matmul(hidden, weights[out]) + biases[out]
    # logits = tf.Print(logits, [weights[fc], biases[fc], weights[out], biases[out]],
    #                   "weights and biases (fc and out) for digit {}".format(i), summarize=10)
    logitss.append(logits)

  return logitss

logitss = setup_nn(tf_train_dataset, weights, biases, train=True)

# losses for weights and biases
loss =  tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logitss[0], tf_train_labels[:, 0]))
# debugging op --
to_print = []
to_print.append(logitss[0])
for i in range(2, NUM_LETTERS+2):
  loss += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logitss[i-1], tf_train_labels[:, i-1]))
  to_print.append(logitss[i-1])

# loss = tf.Print(loss, to_print, "Logits 1 to N", summarize=10)
loss += LAMBDA * tf.nn.l2_loss(weights['fc1'])
loss += LAMBDA * tf.nn.l2_loss(weights['out1'])
loss += LAMBDA * tf.nn.l2_loss(biases['fc1'])
loss += LAMBDA * tf.nn.l2_loss(biases['out1'])

for i in range(2, NUM_LETTERS+2):
  loss += LAMBDA * tf.nn.l2_loss(weights['fc{}'.format(i)])
  loss += LAMBDA * tf.nn.l2_loss(weights['out{}'.format(i)])
  loss += LAMBDA * tf.nn.l2_loss(biases['fc{}'.format(i)])
  loss += LAMBDA * tf.nn.l2_loss(biases['out{}'.format(i)])

# add a summary for loss
tf.scalar_summary('loss', loss)

# Optimizer
optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

# Predictions for the training, validation data
train_prediction = logitss_to_probs(logitss)
valid_logitss = setup_nn(tf_valid_dataset, weights, biases)
valid_prediction = logitss_to_probs(valid_logitss)

# Test data predictions
test_logitss = setup_nn(tf_test_dataset, weights, biases)
test_prediction = logitss_to_probs(test_logitss)

# setup validation loss
tf_valid_labels = tf.constant(valid_labels, dtype=tf.int32)
vloss =  tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(valid_logitss[0], tf_valid_labels[:, 0]))
for i in range(2, NUM_LETTERS+2):
  vloss += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(valid_logitss[i-1], tf_valid_labels[:, i-1]))

tf.scalar_summary('validation loss', vloss)


def accuracy(predictions, labels):
  # predictions is 2-D array of probabilities
  # labels are not one-hot encoded
  predictions = np.array(map(probs_to_labels, predictions)).T
  labels = labels[:, 0:NUM_LETTERS+1].reshape(predictions.shape)
  return (100.0 * np.sum((np.equal(predictions, labels)).all(axis=1))
          / predictions.shape[0])

# Merge all the summaries and write them out to ./logs
session = tf.Session()
merged = tf.merge_all_summaries()
train_writer = tf.train.SummaryWriter('nnlogs' + '/train',
                                      session.graph)

init = tf.initialize_all_variables()
saver = tf.train.Saver()

with session.as_default():
  # Start running the graph operatons
  if not RESTORE:
    session.run(init)
    print("Initialized")
  else:
    saver.restore(session, MODEL_CKPT)
    print("Restored")

  if True:

    # run the training steps if we didn't retore a stored model
    for step in range(NUM_STEPS):

      # Pick an offset within the training data, which has been randomized.
      # Note: we could use better randomization across epochs.
      offset = (step * BATCH_SIZE) % (train_labels.shape[0] - BATCH_SIZE)
      # Generate a minibatch.
      batch_data = train_dataset[offset:(offset + BATCH_SIZE), :]
      batch_labels = train_labels[offset:(offset + BATCH_SIZE), :]
      # Prepare a dictionary telling the session where to feed the minibatch.
      # The key of the dictionary is the placeholder node of the graph to be fed,
      # and the value is the numpy array to feed to it.
      feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
      _, l, predictions = session.run(
        [optimizer, loss, train_prediction], feed_dict=feed_dict)

      if (step % 100 == 0):
        print("Minibatch loss at step %d: %f" % (step, l))
        print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
        valid_predictions = session.run(valid_prediction)
        print("Validation accuracy: %.1f%%" % accuracy(valid_predictions, valid_labels))

        summary = session.run(merged, feed_dict=feed_dict)
        train_writer.add_summary(summary, step)

    # store the model for restoration later
    saved_in = saver.save(session, MODEL_CKPT)
    print("Model saved in {}".format(saved_in))

  # predict the test labels
  test_predictions = session.run(test_prediction)
  print("Test accuracy: %.1f%%" % accuracy(test_predictions, test_labels))

train_writer.close()
