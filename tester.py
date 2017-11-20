import os
import mymnist
import trainer
import tensorflow as tf
import sys
import imageUtil
def test():
  mnist = mymnist.read_data_sets()
  doTest(mnist)

def test(img):
    if(os.path.isfile(img)):
        os.system("cp " + img + " ./tmporiginal.jpg")
    else:
        try:
            os.system("wget -O ./tmporiginal.jpg " + img)
        except:
            print (img + " is not existed or cannot be downloaded")
    imageUtil.resizeSinglePhoto('./tmporiginal.jpg', './tmpresized.jpg')
    mnist = mymnist.read_data_set_one_image('./tmpresized.jpg', one_hot=True)
    doTest(mnist, False)
def doTest(mnist, batch=True):
  x = tf.placeholder(tf.float32, [None, 2352])
  y_ = tf.placeholder(tf.float32, [None, 5])

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

  graph_location = './'
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
    if(batch):
        for i in range(len(mnist.test.images)):
            strresult = strresult + str(prediction.eval(feed_dict={x: [mnist.test.images[i]],keep_prob: 1.0}, session=sess)[0]) + '\n'
        print strresult
        text_file = open(fn, "w")
        text_file.write(strresult)
        text_file.close()
    else:
        res = prediction.eval(feed_dict={x: [mnist.test.images[0]],keep_prob: 1.0}, session=sess)[0];
        print ("single image predicted: " + str(res) + " - " + mymnist.classToFlowerName(res))
    
if __name__ == "__main__":
    if(len(sys.argv) == 1):
        test()
    elif(len(sys.argv) == 2):
        test(sys.argv[1])
