#! /usr/local/bin/python

import sys
import cPickle as pickle
import numpy as np
import tensorflow as tf

# load and format the data
pickle_file = 'SVHN_multi.pickle'

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
NUM_LABELS = 6
BATCH_SIZE = 64
N_HIDDEN_1 = 128
LEARNING_RATE = 0.001
LAMBDA = 0.0001 # regularization rate
NUM_STEPS = 5000
NUM_CHANNELS = 1

def reformat(dataset, labels):
    dataset = dataset.mean(axis=3) # convert to grayscale
    # dataset = dataset.reshape((-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    # labels = (np.arange(1,11) == labels[:,None]).astype(np.float32)
    return dataset, labels

print("After (optional) reformatting ... ")
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

# ***SEEME ***:
# use a small set for validation and test for now
# as the system needs tons of RAM to do convolutions
# on a larger set. We need faster turnaround for now.
valid_dataset = valid_dataset[:200, :]
valid_labels = valid_labels[:200]
test_dataset = test_dataset[:2000, :]
test_labels = test_labels[:2000]

# *** SEEME ***:
# used for validating an architecture
# on a small dataset, if the model overfits to 100% minibatch or training accuracy,
# model is about right and hyperparameter tuning is required.
validate_arch = True
if validate_arch:
    print("Validating architecture")
    train_dataset = train_dataset[:100, :]
    train_labels = train_labels[:100]
    valid_dataset = valid_dataset[:10, :]
    valid_labels = valid_labels[:10]
    test_dataset = test_dataset[:10, :]
    test_labels = test_labels[:10]
    BATCH_SIZE = 10
    #NUM_STEPS = 2000

print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

# global tf computation graph
graph = tf.Graph()

def setup_conv_net(X, weights, biases, train=False):

    # convolution layers with ReLU activations and max pooling
    conv = tf.nn.conv2d(X,
                        weights['conv1'],
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, biases['conv1']))
    #pool = tf.nn.max_pool(relu, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    #print("Pool1 shape: " + str(pool.get_shape().as_list()))

    conv = tf.nn.conv2d(relu,
                        weights['conv2'],
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, biases['conv2']))
    pool = tf.nn.max_pool(relu, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    print("Pool2 shape: " + str(pool.get_shape().as_list()))

    conv = tf.nn.conv2d(pool,
                        weights['conv3'],
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, biases['conv3']))
    pool = tf.nn.max_pool(relu, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    print("Pool3 shape: " + str(pool.get_shape().as_list()))

    # introduce a dropout with probability 0.5 only for training
    # to avoid overfitting.
    if train:
        pool = tf.nn.dropout(pool, 0.5)

    # reshape the resulting cuboid to feed to the
    # downstream fully connected layers
    shape = pool.get_shape().as_list()
    reshape = tf.reshape(pool,
                         [shape[0], shape[1] * shape[2] * shape[3]])

    hidden = tf.nn.relu(tf.matmul(reshape, weights['fc1']) + biases['fc1'])
    logits1 = tf.matmul(hidden, weights['out1']) + biases['out1']

    hidden = tf.nn.relu(tf.matmul(reshape, weights['fc2']) + biases['fc2'])
    logits2 = tf.matmul(hidden, weights['out2']) + biases['out2']

    hidden = tf.nn.relu(tf.matmul(reshape, weights['fc3']) + biases['fc3'])
    logits3 = tf.matmul(hidden, weights['out3']) + biases['out3']

    hidden = tf.nn.relu(tf.matmul(reshape, weights['fc4']) + biases['fc4'])
    logits4 = tf.matmul(hidden, weights['out4']) + biases['out4']

    hidden = tf.nn.relu(tf.matmul(reshape, weights['fc5']) + biases['fc5'])
    logits5 = tf.matmul(hidden, weights['out5']) + biases['out5']

    hidden = tf.nn.relu(tf.matmul(reshape, weights['fc6']) + biases['fc6'])
    logits6 = tf.matmul(hidden, weights['out6']) + biases['out6']

    return (logits1, logits2, logits3, logits4, logits5, logits6)

with graph.as_default():

    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    tf_train_labels = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_LABELS))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Store layers weight & bias
    weights = {
        'conv1': tf.Variable(tf.truncated_normal([5, 5, NUM_CHANNELS, 32], stddev=0.1)), # 5x5 kernel, depth 32
        'conv2': tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1)), # 5x5 kernel, depth 64
        'conv3': tf.Variable(tf.truncated_normal([5, 5, 64, 128], stddev=0.1)), # 5x5 kernel, depth 128
        # after 2 max pooling operations, the feature maps will have 1/(2*2) of the original spatial dimensions
        'fc1': tf.Variable(tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 128, N_HIDDEN_1], stddev=0.1)),
        'out1': tf.Variable(tf.truncated_normal([N_HIDDEN_1, NUM_LABELS], stddev=0.1)),
        'fc2': tf.Variable(tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 128, N_HIDDEN_1], stddev=0.1)),
        'out2': tf.Variable(tf.truncated_normal([N_HIDDEN_1, NUM_LABELS], stddev=0.1)),
        'fc3': tf.Variable(tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 128, N_HIDDEN_1], stddev=0.1)),
        'out3': tf.Variable(tf.truncated_normal([N_HIDDEN_1, NUM_LABELS], stddev=0.1)),
        'fc4': tf.Variable(tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 128, N_HIDDEN_1], stddev=0.1)),
        'out4': tf.Variable(tf.truncated_normal([N_HIDDEN_1, NUM_LABELS], stddev=0.1)),
        'fc5': tf.Variable(tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 128, N_HIDDEN_1], stddev=0.1)),
        'out5': tf.Variable(tf.truncated_normal([N_HIDDEN_1, NUM_LABELS], stddev=0.1)),
        'fc6': tf.Variable(tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 128, N_HIDDEN_1], stddev=0.1)),
        'out6': tf.Variable(tf.truncated_normal([N_HIDDEN_1, NUM_LABELS], stddev=0.1))
        }

    biases = {
        'conv1': tf.Variable(tf.zeros([32])),
        'conv2': tf.Variable(tf.zeros([64])),
        'conv3': tf.Variable(tf.zeros([128])),
        'fc1': tf.Variable(tf.truncated_normal([N_HIDDEN_1])),
        'out1': tf.Variable(tf.truncated_normal([NUM_LABELS])),
        'fc2': tf.Variable(tf.truncated_normal([N_HIDDEN_1])),
        'out2': tf.Variable(tf.truncated_normal([NUM_LABELS])),
        'fc3': tf.Variable(tf.truncated_normal([N_HIDDEN_1])),
        'out3': tf.Variable(tf.truncated_normal([NUM_LABELS])),
        'fc4': tf.Variable(tf.truncated_normal([N_HIDDEN_1])),
        'out4': tf.Variable(tf.truncated_normal([NUM_LABELS])),
        'fc5': tf.Variable(tf.truncated_normal([N_HIDDEN_1])),
        'out5': tf.Variable(tf.truncated_normal([NUM_LABELS])),
        'fc6': tf.Variable(tf.truncated_normal([N_HIDDEN_1])),
        'out6': tf.Variable(tf.truncated_normal([NUM_LABELS]))
        }

    logits1, logits2, logits3, logits4, logits5, logits6 = setup_conv_net(tf_train_dataset, weights, biases, train=True)
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits1, tf_train_labels[0])
        + tf.nn.sparse_softmax_cross_entropy_with_logits(logits2, tf_train_labels[1])
        + tf.nn.sparse_softmax_cross_entropy_with_logits(logits3, tf_train_labels[2])
        + tf.nn.sparse_softmax_cross_entropy_with_logits(logits4, tf_train_labels[3])
        + tf.nn.sparse_softmax_cross_entropy_with_logits(logits5, tf_train_labels[4])
        + tf.nn.sparse_softmax_cross_entropy_with_logits(logits6, tf_train_labels[5])
        + LAMBDA * tf.nn.l2_loss(weights['conv1'])
        + LAMBDA * tf.nn.l2_loss(weights['conv2'])
        + LAMBDA * tf.nn.l2_loss(weights['conv3'])
        + LAMBDA * tf.nn.l2_loss(weights['fc1'])
        + LAMBDA * tf.nn.l2_loss(weights['out1'])
        + LAMBDA * tf.nn.l2_loss(weights['fc2'])
        + LAMBDA * tf.nn.l2_loss(weights['out2'])
        + LAMBDA * tf.nn.l2_loss(weights['fc3'])
        + LAMBDA * tf.nn.l2_loss(weights['out3'])
        + LAMBDA * tf.nn.l2_loss(weights['fc4'])
        + LAMBDA * tf.nn.l2_loss(weights['out4'])
        + LAMBDA * tf.nn.l2_loss(weights['fc5'])
        + LAMBDA * tf.nn.l2_loss(weights['out5'])
        + LAMBDA * tf.nn.l2_loss(weights['fc6'])
        + LAMBDA * tf.nn.l2_loss(weights['out6'])
        + LAMBDA * tf.nn.l2_loss(biases['conv1'])
        + LAMBDA * tf.nn.l2_loss(biases['conv2'])
        + LAMBDA * tf.nn.l2_loss(biases['conv3'])
        + LAMBDA * tf.nn.l2_loss(biases['fc1'])
        + LAMBDA * tf.nn.l2_loss(biases['out1'])
        + LAMBDA * tf.nn.l2_loss(biases['fc2'])
        + LAMBDA * tf.nn.l2_loss(biases['out2'])
        + LAMBDA * tf.nn.l2_loss(biases['fc3'])
        + LAMBDA * tf.nn.l2_loss(biases['out3'])
        + LAMBDA * tf.nn.l2_loss(biases['fc4'])
        + LAMBDA * tf.nn.l2_loss(biases['out4'])
        + LAMBDA * tf.nn.l2_loss(biases['fc5'])
        + LAMBDA * tf.nn.l2_loss(biases['out5'])
        + LAMBDA * tf.nn.l2_loss(biases['fc6'])
        + LAMBDA * tf.nn.l2_loss(biases['out6']))

    # Optimizer.
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_logits = setup_conv_net(tf_valid_dataset, weights, biases)
    valid_prediction = tf.nn.softmax(valid_logits)
    test_logits = setup_conv_net(tf_test_dataset, weights, biases)
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
