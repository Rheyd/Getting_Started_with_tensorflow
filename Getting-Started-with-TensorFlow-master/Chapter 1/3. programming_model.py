#remove the logging info
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#import library 
import tensorflow as tf
#disable eager execution, more like tensorflow 1.14>
tf.compat.v1.disable_eager_execution()

a = tf.compat.v1.placeholder("int32")
b = tf.compat.v1.placeholder("int32")

y = tf.math.multiply(a,b)

sess = tf.compat.v1.Session()

print(sess.run(y, feed_dict={a : 2,b:15}))
