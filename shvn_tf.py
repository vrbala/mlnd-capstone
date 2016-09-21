#! /usr/local/bin/python

import sys
import cPickle as pickle
import numpy as np
import tensorflow as tf

sess = tf.Session()

from keras import backend as K
K.set_session(sess)

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

image_size = 32
def reformat(dataset, labels):
      dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
      # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
      labels = (np.arange(1,11) == labels[:,None]).astype(np.float32)
      return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

# input placeholder
img = tf.placeholder(tf.float32, shape=(None, image_size*image_size))

from keras.layers import Dense
x = Dense(1024, activation='relu')(img)
preds = Dense(10, activation='softmax')(x)

# placeholder for labels
labels = tf.placeholder(tf.float32, shape=(None, 10))

from keras.objectives import categorical_crossentropy
loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
batch_size = 128
with sess.as_default():
    for i in range(1,50):
        offset = (i * batch_size) % (train_labels.shape[0] - batch_size)
        print "Working at offset: {}".format(offset)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        train_step.run(feed_dict={img: batch_data, labels: batch_labels})

from keras.metrics import categorical_accuracy as accuracy
acc_value = accuracy(labels, preds)
with sess.as_default():
    print acc_value.eval(feed_dict={img: test_dataset,
                                    labels: test_labels})
