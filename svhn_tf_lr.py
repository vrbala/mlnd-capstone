#! /usr/local/bin/python

import sys
import cPickle as pickle
import numpy as np
import tensorflow as tf

# load and format the data
pickle_file = 'SVHN.pickle'

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
image_size = 32
num_labels = 10
batch_size = 64
n_hidden_1 = 4096
n_hidden_2 = 2048
learning_rate = 0.001
lambda_ = 0.0001 # regularization rate
num_steps = 10000

def reformat(dataset, labels):
    dataset = dataset.mean(axis=1) # convert to grayscale
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = (np.arange(1,11) == labels[:,None]).astype(np.float32)
    return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

# *** SEEME ***:
# used for debugging a architecture
# on a small dataset, if the model overfits to 100% minibatch or training accuracy,
# model if about right and hyperparameter tuning is required.
# train_dataset = train_dataset[:50, :]
# train_labels = train_labels[:50]
# batch_size = 10

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

# global tf computation graph
graph = tf.Graph()

def setup_nn(X, weights, biases):

    # hidden layers with ReLU units
    wh1 = weights['h1']
    b1 = biases['b1']
    hl1 = tf.nn.relu(tf.matmul(X, wh1) + b1)

    wh2 = weights['h2']
    b2 = biases['b2']
    hl2 = tf.nn.relu(tf.matmul(hl1, wh2) + b2)

    # Training computation.
    w = weights['out']
    b = biases['out']
    logits = tf.matmul(hl2, w) + b # TODO: FIXME

    return logits

with graph.as_default():

    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape=(batch_size, image_size * image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.truncated_normal([image_size*image_size, n_hidden_1])),
        'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.truncated_normal([n_hidden_2, num_labels])) # TODO: FIXME
        }
    biases = {
        'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
        'b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
        'out': tf.Variable(tf.truncated_normal([num_labels]))
        }

    logits = setup_nn(tf_train_dataset, weights, biases)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)
        + lambda_ * tf.nn.l2_loss(weights['h1'])
        + lambda_ * tf.nn.l2_loss(weights['h2']) # TODO: FIXME
        + lambda_ * tf.nn.l2_loss(weights['out']))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_logits = setup_nn(tf_valid_dataset, weights, biases)
    valid_prediction = tf.nn.softmax(valid_logits)
    test_logits = setup_nn(tf_test_dataset, weights, biases)
    test_prediction = tf.nn.softmax(test_logits)

# with graph.as_default():

#   # Input data. For the training data, we use a placeholder that will be fed
#   # at run time with a training minibatch.
#   tf_train_dataset = tf.placeholder(tf.float32,
#                                     shape=(batch_size, image_size * image_size))
#   tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
#   tf_valid_dataset = tf.constant(valid_dataset)
#   tf_test_dataset = tf.constant(test_dataset)

#   hidden_units = 2048

#   # hidden layer with ReLU units
#   weights1 = tf.Variable(
#     tf.truncated_normal([image_size * image_size, hidden_units], stddev=0.35))
#   biases1 = tf.Variable(tf.zeros([hidden_units]))
#   hidden = tf.nn.relu(tf.matmul(tf_train_dataset, weights1) + biases1)

#   weights2 = tf.Variable(
#     tf.truncated_normal([hidden_units, num_labels], stddev=0.35))
#   biases2 = tf.Variable(tf.zeros(num_labels))
#   # Training computation.
#   logits = tf.matmul(hidden, weights2) + biases2

#   loss = tf.reduce_mean(
#     tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

#   # Optimizer.
#   optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#   # Predictions for the training, validation, and test data.
#   train_prediction = tf.nn.softmax(logits)
#   valid_prediction = tf.nn.softmax(tf.matmul(
#     tf.nn.relu(tf.matmul(tf_valid_dataset, weights1) + biases1), weights2) + biases2)
#   test_prediction = tf.nn.softmax(tf.matmul(
#     tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1), weights2) + biases2)

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
