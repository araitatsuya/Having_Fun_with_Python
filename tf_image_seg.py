#
#   Fully Convolutional Network for Semantic Segmentation
#   
#                                   Tatsuya Arai 
#
#   Fully Convolutional Network for Semantic Segmentation in Tensorflow
#   PASCAL VOC Image Segmentation
#   TF-Slim libeary
#   VGG-16 etc.
#
#
#   Python 3.6.1 (Anaconda)
#   tf.__version__ '1.2.0'

#   You can segment any objects in the following categories. 
'''
{0: 'background',
 1: 'aeroplane',
 2: 'bicycle',
 3: 'bird',
 4: 'boat',
 5: 'bottle',
 6: 'bus',
 7: 'car',
 8: 'cat',
 9: 'chair',
 10: 'cow',
 11: 'diningtable',
 12: 'dog',
 13: 'horse',
 14: 'motorbike',
 15: 'person',
 16: 'potted-plant',
 17: 'sheep',
 18: 'sofa',
 19: 'train',
 20: 'tv/monitor',
 255: 'ambigious'}
'''

# Reference
# https://github.com/warmspringwinds/tensorflow_notes/blob/master/fully_convolutional_networks.ipynb
# https://github.com/warmspringwinds/tensorflow_notes

#
# ----tf_image_seg.py
#  |
#  ---models
#  |
#  ---tf_image_segmentation
#  |
#  ---checkpoints
#  |
#  ---Lucy3.jpg (Input Image)
#  |
#  ---Lucy_Sticker.png (Output Image)
#

### Modules
from __future__ import division
import os
import sys
import tensorflow as tf
import skimage.io as io
import numpy as np

# 1st Part
#   Semantic Segmentation
#
#
# You probably need to fix several lines of code. 
# Python 3 related fix
#   urllib2 -> urllib.request
# tensorflow 1.2.0 related fix
#   tensorflow.pack -> tensorflow.stack
#
#

# You need to download models from tensorflow github. 
# https://github.com/tensorflow/models
MODELS = "models/slim"

# You need to download fcn_8s.tar.gz from the following link.
# https://www.dropbox.com/s/7r6lnilgt78ljia/fcn_8s.tar.gz?dl=0
# Unzip tar.gz and place in checkpoints
fcn_8s_checkpoint_path = 'checkpoints/fcn_8s_checkpoint/model_fcn8s_final.ckpt'

sys.path.append(MODELS)

# You don't need CUDA for this.
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

slim = tf.contrib.slim

# 
# You need to download tf-image-segmentation
# https://github.com/warmspringwinds/tf-image-segmentation
#

# from tf_image_segmentation.models.fcn_8s import FCN_8s
sys.path.append("tf-image-segmentation/tf_image_segmentation/models")
from fcn_8s import FCN_8s

#from tf_image_segmentation.utils.inference import adapt_network_for_any_size_input
sys.path.append("tf-image-segmentation/tf_image_segmentation/utils")
from inference import adapt_network_for_any_size_input

from pascal_voc import pascal_segmentation_lut

number_of_classes = 21

image_filename = 'Lucy3.jpg'

image_filename_placeholder = tf.placeholder(tf.string)

feed_dict_to_use = {image_filename_placeholder: image_filename}

image_tensor = tf.read_file(image_filename_placeholder)

image_tensor = tf.image.decode_jpeg(image_tensor, channels=3)

# Fake batch for image and annotation by adding
# leading empty axis.
image_batch_tensor = tf.expand_dims(image_tensor, axis=0)

# Be careful: after adaptation, network returns final labels
# and not logits
FCN_8s = adapt_network_for_any_size_input(FCN_8s, 32)


pred, fcn_8s_variables_mapping = FCN_8s(image_batch_tensor=image_batch_tensor,
                                          number_of_classes=number_of_classes,
                                          is_training=False)

# The op for initializing the variables.
initializer = tf.local_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(initializer)
    # Restore The Checkpoint Model
    saver.restore(sess, fcn_8s_checkpoint_path)
    # Feed Input Image
    image_np, pred_np = sess.run([image_tensor, pred], feed_dict=feed_dict_to_use)
    io.imshow(image_np)
    io.show()
    io.imshow(pred_np.squeeze())
    io.show()

#
# 2nd Part
# Create a contour for the segmentation to make a sticker.  
#   .png takes care of alpha chanel so that the background becomes transparent
#


import skimage.morphology

# Lucy is a cat so that its corresponding category is 8. 
prediction_mask = (pred_np.squeeze() == 8)

# Let's apply some morphological operations to
# create the contour for our sticker

cropped_object = image_np * np.dstack((prediction_mask,) * 3)

square = skimage.morphology.square(5)

temp = skimage.morphology.binary_erosion(prediction_mask, square)

negative_mask = (temp != True)

eroding_countour = negative_mask * prediction_mask

eroding_countour_img = np.dstack((eroding_countour, ) * 3)

cropped_object[eroding_countour_img] = 248

png_transparancy_mask = np.uint8(prediction_mask * 255)

image_shape = cropped_object.shape

png_array = np.zeros(shape=[image_shape[0], image_shape[1], 4], dtype=np.uint8)

png_array[:, :, :3] = cropped_object

png_array[:, :, 3] = png_transparancy_mask

io.imshow(cropped_object)

io.imsave('Lucy_Sticker.png', png_array)
