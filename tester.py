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
  y_conv, keep_prob = trainer.deepnn(x)
  y_prediction = tf.reduce_mean(y_conv, 0)

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
    
    prediction=tf.argmax(y_conv,1)
    print 'length', len(mnist.test.images)
    strresult = ""
    fn = 'project2_20413289.txt'

    try:
        file = open(fn, 'r')
    except:
        file = open(fn, 'w')
        file.close()
    for i in range(len(mnist.test.images)):
        strresult = strresult + str(prediction.eval(feed_dict={x: [mnist.test.images[i]],keep_prob: 1.0}, session=sess)[0]) + '\n'
    print strresult
    text_file = open(fn, "w")
    text_file.write(strresult)
    text_file.close()
          # print mymnist.classToFlowerName(prediction.eval(feed_dict={x: [mnist.test.images[i]],keep_prob: 1.0}, session=sess)[0])
        
    # print prediction.eval(feed_dict={x: [mnist.test.images[4]],keep_prob: 1.0}, session=sess)[0]
    # print prediction.eval(feed_dict={x: [mnist.test.images[400]],keep_prob: 1.0}, session=sess)[0]
    


if __name__ == "__main__":
      test()

