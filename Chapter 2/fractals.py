import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#compatibility mode for tensorflow, runs like a tensorflow 1 no advantages of tf 2.0
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import matplotlib.pyplot as plt 
     

#MANDELBROT SET
#Y, X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]

#JULIA SET
Y, X = np.mgrid[-2:2:0.005, -2:2:0.005]

#Definiamo il punto corrente 
Z = X+1j*Y
Z = tf.constant(Z.astype("complex64"))

zs = tf.Variable(Z)
ns = tf.Variable(tf.zeros_like(Z, "float32"))


sess = tf.InteractiveSession()
tf.initialize_all_variables().run()
c = complex(0.0,0.75)
#c = complex(-1.5,-1.5)


# Compute the new values of z: z^2 + x
#zs_ = zs*zs + c
zs_ = zs*zs - c

# Have we diverged with this new value?
not_diverged = tf.abs(zs_) < 4

step = tf.group(
  zs.assign(zs_),
  ns.assign_add(tf.cast(not_diverged, "float32"))
  )

for i in range(200): step.run()

plt.imshow(ns.eval())
plt.show()