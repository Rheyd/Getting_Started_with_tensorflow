import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#compatibility mode for tensorflow, runs like a tensorflow 1 no advantages of tf 2.0
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x = tf.placeholder(tf.float32)
func =  2*x*x   
var_grad = tf.gradients(func, x)
with tf.Session() as session:
    var_grad_val = session.run(var_grad,feed_dict={x:1})
    print(var_grad_val)
    


