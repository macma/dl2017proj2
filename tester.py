import mymnist
import trainer
import tensorflow as tf

def test():
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
    saver = tf.train.Saver()
    saver.restore(sess, "./model/ckp")
    #saver = tf.train.import_meta_graph("./model/ckp.meta")

    # saver.save(sess,"./model/ckp")
    #print(sess.run(y_,{x: batch[0]}))
    ret = sess.run(y_conv, feed_dict = {x : mnist.test.images})
    print("result predicted:%d"%(ret))
    # prediction=tf.argmax(y_conv,1)
    # print(sess.run(y_conv, feed_dict={x: mnist.test.images[0].reshape(1,2352)}))
    # prediction=tf.argmax(y_conv,1)
    # print "predictions", y_conv.eval(feed_dict={x: mnist.test.images}, session=sess)
    # classification = sess.run(y_, feed_dict)
    # print 'aaaaaa', y_conv.eval()
    print('test accuracy %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

test()
# run()

