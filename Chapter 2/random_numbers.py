import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#compatibility mode for tensorflow, runs like a tensorflow 1 no advantages of tf 2.0
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt



uniform = tf.random_uniform([100],minval=0,maxval=1,dtype=tf.float32)
with tf.Session() as session:
    print (uniform.eval())
    plt.hist(uniform.eval(),normed=True)
    plt.show() 

# Create a tensor of shape [100] consisting of random normal values, with mean
# 0 and standard deviation 2.
norm = tf.random_normal([100], mean=0, stddev=2)
with tf.Session() as session:
    plt.hist(norm.eval(),normed=True)
    plt.show()  

uniform_with_seed = tf.random_uniform([1], seed=1)
uniform_without_seed = tf.random_uniform([1])

# Repeatedly running this block with the same graph will generate the same
# sequence of values for 'a', but different sequences of values for 'b'.
print("First Run")
with tf.Session() as first_session:
  print("uniform with (seed = 1) = {}"\
        .format(first_session.run(uniform_with_seed)))  
  print("uniform with (seed = 1) = {}"\
        .format(first_session.run(uniform_with_seed)))
  print("uniform without seed = {}"\
        .format(first_session.run(uniform_without_seed)))  
  print("uniform without seed = {}"\
        .format(first_session.run(uniform_without_seed)))  

print("Second Run")
with tf.Session() as second_session:
  print("uniform with (seed = 1) = {}"\
        .format(second_session.run(uniform_with_seed)))  
  print("uniform with (seed = 1) = {}"\
        .format(second_session.run(uniform_with_seed)))  
  print("uniform without seed = {}"\
        .format(second_session.run(uniform_without_seed)))  
  print("uniform without seed = {}"\
        .format(second_session.run(uniform_without_seed)))
