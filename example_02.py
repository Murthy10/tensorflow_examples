import tensorflow as tf

with tf.device('/cpu:0'): #'/gpu:0'
    a = tf.constant(6.0, dtype='float', shape=[1], name='a')
    b = tf.constant(7.0, dtype='float', shape=[1], name='b')
    c = tf.mul(a, b)

with tf.Session() as sess:
    result = sess.run(c)
    print(result)
