#! /usr/local/bin/python

import os

from scipy import misc

# Typical setup to include TensorFlow.
import tensorflow as tf

root = 'train'
print "Working on {}".format(root)

num_of_files = len(os.listdir("{}/proc/".format(root)))
print "Processing {} files. ".format(num_of_files)

# Make a queue of file names including all the PNG images files in the relative
# image directory.
filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once("{}/proc/*.png".format(root)))

# reader to split up the file.
image_reader = tf.WholeFileReader()

# image_size - 32x32 images
IMAGE_SIZE = 32

# Read a whole file from the queue, the first returned value in the tuple is the
# filename which we are ignoring.
filename, image_file = image_reader.read(filename_queue)

# Decode the image as a JPEG file, this will turn it into a Tensor which we can
# then use in training.
image = tf.image.decode_png(image_file)

# resize the image
resized_image = tf.image.resize_images(image, IMAGE_SIZE, IMAGE_SIZE)

# Start a new session to show example output.
with tf.Session() as sess:

    # Required to get the filename matching to run.
    tf.initialize_all_variables().run()
    im_size = tf.constant([IMAGE_SIZE, IMAGE_SIZE])

    # Coordinate the loading of image files.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        for _ in range(num_of_files):
            # Get an image tensor and print its value.
            image_file, image_tensor = sess.run([filename, resized_image])
            print 'Processing ', image_file
            misc.imsave('{}/proc/resized/{}'.format(root, os.path.basename(image_file)), image_tensor)

    except tf.errors.OutOfRangeError:
        print('Done -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    coord.join(threads)
