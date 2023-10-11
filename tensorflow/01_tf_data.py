import tensorflow as tf
import numpy as np

# (2, 2, 3)
d3_tensor = tf.constant([[[1, 2, 3],
                         [4, 5, 6]],
                         [[-1, -2, -3],
                          [-4, -5, -6]]])

d3_tensor[0, 0, 0]
d3_tensor.shape
d3_tensor.dtype
np_d3_tensor = d3_tensor.numpy()
np_d3_tensor.shape
np_d3_tensor.dtype
np_d3_tensor[0, 0, 0] = 0

d3_tensor.ndim
d3_tensor.shape
d3_tensor.dtype

d3_np_tensor = np.array([[[1, 2, 3],
                         [4, 5, 6]],
                         [[-1, -2, -3],
                          [-4, -5, -6]]])
d3_np_tensor = d3_np_tensor.astype(np.float32)
d3_var_tensor = tf.Variable(d3_np_tensor)
d3_var_tensor[0, 0, 0].assign(0)