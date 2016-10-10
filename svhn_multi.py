#! /usr/local/bin/python

import sys
import math
import cPickle as pickle
import numpy as np
from scipy import misc
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
NUM_LABELS = 10 # digits 0-9
BATCH_SIZE = 1
N_HIDDEN_1 = 128
LEARNING_RATE = 0.00001
LAMBDA = 0.00001 # regularization rate
NUM_STEPS = 5000
NUM_CHANNELS = 1
# number of letters in the sequence to transcribe
NUM_LETTERS = 1

def reformat(dataset, labels):
    dataset = dataset.reshape(-1, 3, IMAGE_SIZE, IMAGE_SIZE)
    dataset = dataset.mean(axis=1) # convert to grayscale
    dataset = dataset.reshape((-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    # labels = (np.arange(1,11) == labels[:,None]).astype(np.float32)
    return dataset, labels

print("After (optional) reformatting ... ")
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

# *** SEEME ***
# weedout the bad images
# they cause NaN loss function and thus NaN weights
# need to see why those images occur
# train_dataset = np.delete(train_dataset, (794), axis=0)
# train_labels = np.delete(train_labels, (794), axis=0)

# *** SEEME ***:
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
validate_arch = False
if validate_arch:
    print("Validating architecture")
    train_dataset = train_dataset[:100, :]
    train_labels = train_labels[:100]
    valid_dataset = valid_dataset[:10, :]
    valid_labels = valid_labels[:10]
    test_dataset = test_dataset[:10, :]
    test_labels = test_labels[:10]
    BATCH_SIZE = 16
    NUM_STEPS = 1000

print('Inputs to the model')
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

# global tf computation graph
graph = tf.Graph()

def setup_conv_net(X, weights, biases, train=False):

    to_print = []
    # convolution layers with ReLU activations and max pooling
    conv = tf.nn.conv2d(X,
                        weights['conv1'],
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, biases['conv1']))
    #pool = tf.nn.max_pool(relu, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    #print("Pool1 shape: " + str(pool.get_shape().as_list()))

    to_print.append(weights['conv1'])
    to_print.append(conv)
    to_print.append(relu)

    conv = tf.nn.conv2d(relu,
                        weights['conv2'],
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, biases['conv2']))
    pool = tf.nn.max_pool(relu, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    print("Pool2 shape: " + str(pool.get_shape().as_list()))

    to_print.append(weights['conv2'])
    to_print.append(conv)
    to_print.append(relu)
    to_print.append(pool)

    conv = tf.nn.conv2d(pool,
                        weights['conv3'],
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, biases['conv3']))
    pool = tf.nn.max_pool(relu, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    print("Pool3 shape: " + str(pool.get_shape().as_list()))

    to_print.append(weights['conv3'])
    to_print.append(conv)
    to_print.append(relu)
    to_print.append(pool)

    # introduce a dropout with probability 0.5 only for training
    # to avoid overfitting.
    if False:
        pool = tf.nn.dropout(pool, 0.5)

    # reshape the resulting cuboid to feed to the
    # downstream fully connected layers
    shape = pool.get_shape().as_list()
    reshape = tf.reshape(pool,
                         [shape[0], shape[1] * shape[2] * shape[3]])

    logitss = []
    hidden = tf.nn.relu(tf.matmul(reshape, weights['fc1']) + biases['fc1'])
    logits = tf.matmul(hidden, weights['out1']) + biases['out1']
    to_print.append(weights['fc1'])
    to_print.append(biases['fc1'])
    to_print.append(weights['out1'])
    to_print.append(biases['out1'])
    to_print.append(logits)
    # logits = tf.Print(logits, to_print,
    #                             "conv, relu, conv, relu, pool, conv, relu, pool, W & b(fc1 and out1), logits\n",
    #                             summarize=5)
    logitss.append(logits)

    for i in range(2, NUM_LETTERS+2):
        hidden = tf.nn.relu(tf.matmul(reshape, weights['fc{}'.format(i)]) + biases['fc{}'.format(i)])
        logitss.append(tf.matmul(hidden, weights['out{}'.format(i)]) + biases['out{}'.format(i)])

    return logitss

def probs_to_labels(probs):
    # input: 2-D array of probabilities (result of softmax)
    # output: a list of labels of size probs.shape[0]
   return [np.argmax(x) for x in probs]

def logitss_to_probs(logitss):
    # input: a list of logits
    # output: a 2-D array of softmax operations (they have to be eval'ed in tf session)
    # just applies softmax on each of the logits
    return map(tf.nn.softmax, logitss)

with graph.as_default():

    tf.set_random_seed(4096)

    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    tf_train_labels = tf.placeholder(tf.int32, shape=(BATCH_SIZE, 6)) # 6 here is 1 digit for length of sequence and 5 for digits themselves
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)

    # Store layers weight & bias
    # after 2 max pooling operations, the feature maps will have 1/(2*2) of the original spatial dimensions
    weights = {
        'conv1': tf.Variable(tf.truncated_normal([5, 5, NUM_CHANNELS, 32], stddev=2.0/math.sqrt(5*5*NUM_CHANNELS*32))), # 5x5 kernel, depth 32
        'conv2': tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=2.0/(5*5*32*64))), # 5x5 kernel, depth 64
        'conv3': tf.Variable(tf.truncated_normal([5, 5, 64, 128], stddev=2.0/(math.sqrt(5*5*64*128)))), # 5x5 kernel, depth 128
        # for the length of the sequence of digits
        'fc1': tf.Variable(tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 128, N_HIDDEN_1], stddev=2.0/math.sqrt(IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 128))),
        'out1': tf.Variable(tf.truncated_normal([N_HIDDEN_1, 5], stddev=2.0/math.sqrt(N_HIDDEN_1))), # length of the sequence: here 1-5 - TODO: make it configurable
        }

    # for individual digits
    for i in range(2, NUM_LETTERS+2):
        weights['fc{}'.format(i)] = tf.Variable(tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 128, N_HIDDEN_1], stddev=2.0/math.sqrt(IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 128)))
        weights['out{}'.format(i)] = tf.Variable(tf.truncated_normal([N_HIDDEN_1, NUM_LABELS], stddev=2.0/math.sqrt(N_HIDDEN_1)))

    biases = {
        'conv1': tf.Variable(tf.zeros([32])),
        'conv2': tf.Variable(tf.zeros([64])),
        'conv3': tf.Variable(tf.zeros([128])),
        # for the length of sequence: here 1-5
        'fc1': tf.Variable(tf.truncated_normal([N_HIDDEN_1])),
        'out1': tf.Variable(tf.truncated_normal([5])),
        }

    # for individual digits
    for i in range(2, NUM_LETTERS+2):
        biases['fc{}'.format(i)] = tf.Variable(tf.truncated_normal([N_HIDDEN_1]))
        biases['out{}'.format(i)] = tf.Variable(tf.truncated_normal([NUM_LABELS]))

    logitss = setup_conv_net(tf_train_dataset, weights, biases, train=True)

    # losses for weights and biases
    loss =  tf.nn.sparse_softmax_cross_entropy_with_logits(logitss[0], tf_train_labels[:, 0])
    for i in range(2, NUM_LETTERS+2):
        loss += tf.nn.sparse_softmax_cross_entropy_with_logits(logitss[i-1], tf_train_labels[:, i-1])

    loss += LAMBDA * tf.nn.l2_loss(weights['conv1'])
    loss += LAMBDA * tf.nn.l2_loss(weights['conv2'])
    loss += LAMBDA * tf.nn.l2_loss(weights['conv3'])
    loss += LAMBDA * tf.nn.l2_loss(biases['conv1'])
    loss += LAMBDA * tf.nn.l2_loss(biases['conv2'])
    loss += LAMBDA * tf.nn.l2_loss(biases['conv3'])
    loss += LAMBDA * tf.nn.l2_loss(weights['fc1'])
    loss += LAMBDA * tf.nn.l2_loss(weights['out1'])
    loss += LAMBDA * tf.nn.l2_loss(biases['fc1'])
    loss += LAMBDA * tf.nn.l2_loss(biases['out1'])

    for i in range(2, NUM_LETTERS+2):
        loss += LAMBDA * tf.nn.l2_loss(weights['fc{}'.format(i)])
        loss += LAMBDA * tf.nn.l2_loss(weights['out{}'.format(i)])
        loss += LAMBDA * tf.nn.l2_loss(biases['fc{}'.format(i)])
        loss += LAMBDA * tf.nn.l2_loss(biases['out{}'.format(i)])

    # loss is 1-D array of size - batch_size
    loss = tf.reduce_mean(loss)

    # Optimizer.
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = logitss_to_probs(logitss)
    valid_logitss = setup_conv_net(tf_valid_dataset, weights, biases)
    valid_prediction = logitss_to_probs(valid_logitss)
    test_logitss = setup_conv_net(tf_test_dataset, weights, biases)
    test_prediction = logitss_to_probs(test_logitss)

def accuracy(predictions, labels):
    # predictions is 2-D array of probabilities
    # labels are not one-hot encoded
    predictions = np.array(map(probs_to_labels, predictions)).T
    labels = labels[:, 0:NUM_LETTERS+1].reshape(predictions.shape)
    return (100.0 * np.sum((np.equal(predictions, labels)).all(axis=1))
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

    try:
      if np.isnan(l):
        print("Step: {}".format(step))
        # if we get, NaN loss, write out the images to tmp dir for analysis and exit
        for i, im in enumerate(batch_data):
          im = im.reshape(IMAGE_SIZE, IMAGE_SIZE)
          print(im)
          misc.imsave('tmp/bad_image{}.png'.format(i), im)
        raise RuntimeError
    except:
      sys.exit(0)

    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      valid_predictions = session.run(valid_prediction)
      print("Validation accuracy: %.1f%%" % accuracy(valid_predictions, valid_labels))

  test_predictions = session.run(test_prediction)
  print("Test accuracy: %.1f%%" % accuracy(test_predictions, test_labels))
