import tensorflow as tf
import numpy as np

# constant tensor
tensor_a = tf.reshape(tf.range(1, 7, dtype=tf.float32), (2, 3))
tensor_b = tf.reshape(tf.range(7, 13, dtype=tf.float32), (2, 3))

tensor_c = tensor_a + tensor_b
tensor_d = tf.math.square(tensor_c)
tensor_e = tf.math.log(tensor_d)

with tf.GradientTape() as tape:
    tape.watch(tensor_a)
    tensor_c = tf.square(tensor_a + tensor_b)
    gradient_ca = tape.gradient(target=tensor_c, sources=tensor_a)
    print(gradient_ca)

# variable
tensor_var_a = tf.Variable(tensor_a)
tensor_var_b = tf.Variable(tensor_b)

with tf.GradientTape() as tape:
    tensor_var_c = tf.square(tensor_var_a + tensor_var_b)
    gradient_ca = tape.gradient(target=tensor_var_c, sources=tensor_var_a)
    print(gradient_ca)