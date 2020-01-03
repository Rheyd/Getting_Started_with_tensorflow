#handling tensors


#STEP 1 --- PREPARE THE DATA
import matplotlib.image as mp_image
filename = "packt.jpeg"
input_image = mp_image.imread(filename)

#dimension
print('input dim = {}'.format(input_image.ndim))
#shape
print('input shape = {}'.format(input_image.shape))

import matplotlib.pyplot as plt
plt.imshow(input_image)
plt.show()

import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#compatibility mode for tensorflow, runs like a tensorflow 1 no advantages of tf 2.0
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

my_image = tf.placeholder("uint8",[None,None,3])
slice = tf.slice(my_image,[10,0,0],[16,-1,-1])


with tf.Session() as session:
    result = session.run(slice,feed_dict={my_image: input_image})
    print(result.shape)

plt.imshow(result)
plt.show()
