import os
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import misc

root = 'train'
print "Working on {}".format(root)
fileName = '{}/digitStruct.csv'.format(root)

df = pd.read_csv(fileName)
df['LabelCount'] = df.DigitLabel
# right and bottom most offsets
df['RightMost'] = df.Left + df.Width
df['BottomMost'] = df.Top + df.Height

grouped = df.groupby(['FileName'])
agg = grouped.agg({'Left': np.min, 'Top': np.min, 'BottomMost': np.max, 'RightMost': np.max, 'LabelCount': np.size, 'DigitLabel': lambda x: tuple(x)})
agg['Height'] = agg.BottomMost - agg.Top
agg['Width'] = agg.RightMost - agg.Left

session = tf.Session()
with session.as_default():
    tf.initialize_all_variables().run()
    for name, data in agg.iterrows():
        im = tf.constant(misc.imread('{}/{}'.format(root, name)))
        im_height, im_width, _ = im.get_shape().as_list()
        offset_height = data['Top']
        offset_width = data['Left']
        target_height = data['Height']
        target_width = data['Width']
        # edge cases - bounding box is at the boundary of the image
        if offset_height + target_height > im_height:
            target_height = im_height - offset_height
        if offset_width + target_width > im_width:
            target_width = im_width - offset_width
        print "{}: (oh: {} ow: {} th: {} tw: {} image dims: {})".format(name, offset_height, offset_width, target_height, target_width, im.get_shape().as_list())
        try:
            x = tf.image.crop_to_bounding_box(im, offset_height, offset_width, target_height, target_width)
        except ValueError:
            print "Bad bounding box for {}".format(name)
            continue
        x = session.run(x)
        print x.shape
        misc.imsave('{}/proc/{}'.format(root, name), x)
