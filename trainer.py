# https://stackoverflow.com/questions/37340129/tensorflow-training-on-my-own-image
# http://ischlag.github.io/2016/06/19/tensorflow-input-pipeline-example/
import tensorflow as tf
import os
import mymnist
# step 1
# filenames = ['test_photos/100080576_f52e8ee070_n.jpg', 'test_photos/10140303196_b88d3d6cec.jpg',
#              'test_photos/10172379554_b296050f82_n.jpg', 'test_photos/10172567486_2748826a8b.jpg']
# 10172636503_21bededa75_n
# filenames = ['test_photos/102501987_3cdb8e5394_n.jpg', 'test_photos/10503217854_e66a804309.jpg',
#              'test_photos/10894627425_ec76bbc757_n.jpg', 'test_photos/110472418_87b6a3aa98_m.jpg']
# 11102341464_508d558dfc_n
#
# step 2
# filename_queue = tf.train.string_input_producer(filenames)
#
# step 3: read, decode and resize images
# reader = tf.WholeFileReader()
# filename, content = reader.read(filename_queue)
# image = tf.image.decode_jpeg(content, channels=3)
# image = tf.cast(image, tf.float32)
# resized_image = tf.image.resize_images(image, [128, 128])
#
# step 4: Batching
# image_batch = tf.train.batch([resized_image], batch_size=8)

resizedir = "./resized_photos28"


def dirToClass(flowername):
    if(flowername == 'tulips'):
        return 1;
    if(flowername == 'sunflowers'):
        return 2;
    if(flowername == 'roses'):
        return 3;
    if(flowername == 'dandelion'):
        return 4;
    if(flowername == 'daisy'):
        return 5;
# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.


def _parse_function(filename, label):
    print ('dddd')
    print (filename, label)
    image_string = tf.read_file(filename)
    print ('eee')
    image_decoded = tf.image.decode_image(image_string)
    print ('fff', image_decoded)
#     image_resized = tf.image.resize_images(image_decoded, [28, 28])

    print ('ggg')
    return image_decoded, label


batch_start = 0


def next_batch(batch_size, batch_data):
      print (type(batch_data))
      global batch_start
      temp_start = batch_start;
      batch_start = temp_start + batch_size
      end = temp_start + batch_size
      if(end > len(batch_data)):
            end = len(batch_data)
      return batch_data[temp_start, end]
    #   shuffle = True
    #   start = self._index_in_epoch
    # # Shuffle for the first epoch
    # if _epochs_completed == 0 and start == 0 and shuffle:
    #   perm0 = numpy.arange(self._num_examples)
    #   numpy.random.shuffle(perm0)
    #   self._images = self.images[perm0]
    #   self._labels = self.labels[perm0]
    # # Go to the next epoch
    # if start + batch_size > self._num_examples:
    #   # Finished epoch
    #   _epochs_completed += 1
    #   # Get the rest examples in this epoch
    #   rest_num_examples = self._num_examples - start
    #   images_rest_part = self._images[start:self._num_examples]
    #   labels_rest_part = self._labels[start:self._num_examples]
    #   # Shuffle the data
    #   if shuffle:
    #     perm = numpy.arange(self._num_examples)
    #     numpy.random.shuffle(perm)
    #     self._images = self.images[perm]
    #     self._labels = self.labels[perm]
    #   # Start next epoch
    #   start = 0
    #   self._index_in_epoch = batch_size - rest_num_examples
    #   end = self._index_in_epoch
    #   images_new_part = self._images[start:end]
    #   labels_new_part = self._labels[start:end]
    #   return numpy.concatenate((images_rest_part, images_new_part), axis=0) , numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
    # else:
    #   self._index_in_epoch += batch_size
    #   end = self._index_in_epoch
    #   return self._images[start:end], self._labels[start:end]


def batchImages():
    fn = []
    lb = []
    counter = 0;
    for subdir, dirs, files in os.walk(resizedir):
        for dir in dirs:
            if(dir != 'test'):
                for subdir1, dirs1, files1 in os.walk(resizedir + "/" + dir):
                    for file in files1:
                        fn.append(resizedir + "/" + dir + "/" + file)
                        lb.append(dirToClass(dir))
                        counter = counter + 1
    # A vector of filenames.
    filenames = tf.constant(fn)
    # `labels[i]` is the label for the image in `filenames[i].
    labels = tf.constant(lb)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(_parse_function)
    return dataset, labels


def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.
  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 3])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, 3, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, 5])
    b_fc2 = bias_variable([5])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def run():
      # Import data
  mnist = mymnist.read_data_sets()

  # Create the model
  x = tf.placeholder(tf.float32, [None, 2352])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 5])

  # Build the graph for the deep net
  y_conv, keep_prob = deepnn(x)

  with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  graph_location = './'#tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
      batch = mnist.train.next_batch(50)
      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
# def run():
#   # Import data
#   #mnist, labels = batchImages();
#   mnist = mymnist.read_data_sets()
#   #mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

#   # Create the model
#   x = tf.placeholder(tf.float32, [None, 784])

#   # Define loss and optimizer
#   y_ = tf.placeholder(tf.float32, [None, 10])

#   # Build the graph for the deep net
#   y_conv, keep_prob = deepnn(x)

#   with tf.name_scope('loss'):
#     cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
#                                                             logits=y_conv)
#   cross_entropy = tf.reduce_mean(cross_entropy)

#   with tf.name_scope('adam_optimizer'):
#     train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#   with tf.name_scope('accuracy'):
#     correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
#     correct_prediction = tf.cast(correct_prediction, tf.float32)
#   accuracy = tf.reduce_mean(correct_prediction)

#   graph_location = './'
#   #tempfile.mkdtemp()
#   print('Saving graph to: %s' % graph_location)
#   train_writer = tf.summary.FileWriter(graph_location)
#   train_writer.add_graph(tf.get_default_graph())

#   with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for i in range(20000):
#           batch = mnist.train.next_batch(50)
#       if i % 100 == 0:
#             train_accuracy = accuracy.eval(feed_dict={
#               x: batch[0], y_: batch[1], keep_prob: 1.0})
#             print('step %d, training accuracy %g' % (i, train_accuracy))
#       train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
#     print('test accuracy %g' % accuracy.eval(feed_dict={
#         x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
run()