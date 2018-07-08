import tensorflow as tf
import numpy as np

session = tf.Session()

zeroD = np.array(30, dtype=np.int32)
print("ZeroD rank: ", session.run(tf.rank(zeroD)))
print("ZeroD shape: ", session.run(tf.shape(zeroD)))

oneD = np.array([1.2, 3.5, 6.8, 9.0], dtype=np.float32)
print("OneD rank: ", session.run(tf.rank(oneD)))
print("OneD shape: ", session.run(tf.rank(zeroD)))

session.close()
