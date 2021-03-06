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
IMAGE_SIZE = 32
NUM_LABELS = 10
BATCH_SIZE = 64
N_HIDDEN_1 = 256
N_HIDDEN_2 = 128
LEARNING_RATE = 0.3
LAMBDA = 0.00001 # regularization rate
NUM_STEPS = 10000

def reformat(dataset, labels):
    dataset = dataset.mean(axis=1) # convert to grayscale
    dataset = dataset.reshape((-1, IMAGE_SIZE * IMAGE_SIZE)).astype(np.float32)
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
# BATCH_SIZE = 10

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
    logits = tf.matmul(hl1, w) + b # TODO: FIXME

    return logits

with graph.as_default():

    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape=(BATCH_SIZE, IMAGE_SIZE * IMAGE_SIZE))
    tf_train_labels = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_LABELS))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.truncated_normal([IMAGE_SIZE*IMAGE_SIZE, N_HIDDEN_1])),
        'h2': tf.Variable(tf.truncated_normal([N_HIDDEN_1, N_HIDDEN_2])),
        'out': tf.Variable(tf.truncated_normal([N_HIDDEN_1, NUM_LABELS])) # TODO: FIXME
        }
    biases = {
        'b1': tf.Variable(tf.truncated_normal([N_HIDDEN_1])),
        'b2': tf.Variable(tf.truncated_normal([N_HIDDEN_2])),
        'out': tf.Variable(tf.truncated_normal([NUM_LABELS]))
        }

    logits = setup_nn(tf_train_dataset, weights, biases)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)
        + LAMBDA * tf.nn.l2_loss(weights['h1'])
    #            + LAMBDA * tf.nn.l2_loss(weights['h2']) # TODO: FIXME
        + LAMBDA * tf.nn.l2_loss(weights['out'])
        + LAMBDA * tf.nn.l2_loss(biases['b1'])
    #       + LAMBDA * tf.nn.l2_loss(biases['b2'])
        + LAMBDA * tf.nn.l2_loss(biases['out']))

    batch = tf.Variable(0.0)
    # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE,                # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        train_dataset.shape[0],          # Decay step.
        0.95,                # Decay rate.
        staircase=True)

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_logits = setup_nn(tf_valid_dataset, weights, biases)
    valid_prediction = tf.nn.softmax(valid_logits)
    test_logits = setup_nn(tf_test_dataset, weights, biases)
    test_prediction = tf.nn.softmax(test_logits)


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print("Initialized")
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
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
