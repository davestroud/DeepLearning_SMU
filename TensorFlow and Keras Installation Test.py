# -*- coding: utf-8 -*-
"""
TensorFlow and Keras installation test
"""

import tensorflow as tf
import keras.backend as K

#create message
message = tf.constant('Hello world!')

with tf.Session() as session:
    #Print 'Hello world!'
    session.run(message)
    print(message.eval())
    #List devices tensorflow sees
    devices = session.list_devices()
    for d in devices:
        print('\n\n', d.name)
        
    with tf.device('/GPU:0'):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
        print('\n\n', a.shape, b.shape)
        c = tf.matmul(a, b)

        # with tf.Session() as sess:
        print('\n\n', session.run(c))

#verify Keras is working   
print('\n\nEpsilon:', K.epsilon())

