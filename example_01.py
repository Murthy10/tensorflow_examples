import tensorflow as tf

a = tf.constant(40.0, dtype='float', shape=[1], name='a')
b = tf.constant(2.0, dtype='float', shape=[1], name='b')
c = tf.add(a, b)

with tf.Session() as sess:
    result = sess.run(c)
    print(result)
