
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import os
# import utils.fileUtil as file
import base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from PIL import Image
# from utils.labelFile2Map import *


def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


resizedir = "./resized_photos28"


def dirToClass(flowername):
    if(flowername == 'daisy'):
        return 0
    if(flowername == 'dandelion'):
        return 1
    if(flowername == 'roses'):
        return 2
    if(flowername == 'sunflowers'):
        return 3
    if(flowername == 'tulips'):
        return 4
    return 0


def classToFlowerName(cls):
    if(cls == 0):
        return 'daisy'
    if(cls == 1):
        return 'dandelion'
    if(cls == 2):
        return 'roses'
    if(cls == 3):
        return 'sunflowers'
    if(cls == 4):
        return 'tulips'


def process_images(label_file, one_hot=True, num_classes=5):
    filename = ""
    if label_file == 1:
        filename = "train.txt"
    if label_file == 2:
        filename = "val.txt"
    if label_file == 3:
        filename = "test.txt"

    file = open(filename, "r")
    str = file.read()
    counter = 0
    strlist = str.strip().split('\n')
    images = numpy.empty((len(strlist), 2352))
    labels = numpy.zeros(len(strlist))
    lines = len(strlist)
    for i in range(len(strlist)):
        image = Image.open(strlist[i].split(' ')[0].replace("flower_photos", "resized_photos28"));

        img_ndarray = numpy.asarray(image, dtype='float32')
        images[i] = numpy.ndarray.flatten(img_ndarray)
        if(label_file <> 3):
            labels[i] = int(strlist[i].split(' ')[1])
    rows = 28
    cols = 28
    return images.reshape(lines, rows, cols, 3), dense_to_one_hot(numpy.array(labels, dtype=numpy.uint8), num_classes)
    
    
def process_image(img, one_hot=True, num_classes=5):
    image = Image.open(img)
    img_ndarray = numpy.asarray(image, dtype='float32')
    
    images = numpy.empty((1, 2352))
    labels = numpy.zeros(1)
    images[0] = numpy.ndarray.flatten(img_ndarray)
    # labels[0] = 0
    rows = 28
    cols = 28
    return images.reshape(1, rows, cols, 3), dense_to_one_hot(numpy.array(labels, dtype=numpy.uint8), num_classes)


def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  print ('in onehot', labels_dense, num_classes)
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


def read_data_sets(one_hot=False,
                   dtype=dtypes.float32,
                   reshape=True,
                   validation_size=500,
                   seed=None):
    train_images, train_labels = process_images(1, one_hot=one_hot)
    validation_images, validation_labels = process_images(2, one_hot=one_hot)
    test_images, test_labels = process_images(3, one_hot=one_hot)
    print ('test_images', len(train_images), '-', len (validation_images), '--,', len(test_images))
    
    train = DataSet(
        train_images, train_labels, dtype=dtype, reshape=reshape, seed=seed)
    validation = DataSet(
        validation_images,
        validation_labels,
        dtype=dtype,
        reshape=reshape,
        seed=seed)
    test = DataSet(
        test_images, test_labels, dtype=dtype, reshape=reshape, seed=seed)

    return base.Datasets(train=train, validation=validation, test=test)
    
    

def read_data_set_one_image(img, one_hot=False,
                   reshape=True,
                   dtype=dtypes.float32,
                   seed=None):
    test_images, test_labels = process_image(img, one_hot=one_hot)
    test = DataSet(
        test_images, test_labels, dtype=dtype, reshape=reshape, seed=seed)

    return base.Datasets(test=test, train = test, validation = test)
    
    

class DataSet(object):
  def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True,
               seed=None):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.  Seed arg provides for convenient deterministic testing.
    """
    seed1, seed2 = random_seed.get_seed(seed)
    # If op level seed is not set, use whatever graph level seed is returned
    numpy.random.seed(seed1 if seed is None else seed2)
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      if reshape:
        assert images.shape[3] == 3
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2] * 3)
      if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = numpy.arange(self._num_examples)
        numpy.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return numpy.concatenate((images_rest_part, images_new_part), axis=0), numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]


def load_mnist(train_dir='MNIST-data'):
  return read_data_sets()


if __name__ == "__main__":
  data_dir = "../MNIST_data/"
  read_data_sets(one_hot=True)
