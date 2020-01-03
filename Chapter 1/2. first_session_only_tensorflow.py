#remove the logging info
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#import tensorflow library
import tensorflow as tf

# Construct a `Session` to execute the graph, but in compatibility mode
with tf.compat.v1.Session() as sess:
    # Build a graph.
    x = tf.constant(1,name='x')
    y = tf.Variable(x+9,name='y')

	#initialize variables but in compatibility mode
    model=tf.compat.v1.global_variables_initializer()

	#run the variabels initialization
    sess.run(model)

    #run the variables
    print(sess.run(y))