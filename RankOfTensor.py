import tensorflow as tf


session = tf.Session()

#zero dimension
zeroD = tf.constant(5)
print("Rank : ", session.run(tf.rank(zeroD)))

#one dimension
oneD = tf.constant(["How", "Are", "You"])
print("Rank : ", session.run(tf.rank(oneD)))

#two dimension
twoD = tf.constant([[1, 3], [3, 4], [6, 7]])
print("Rank : ", session.run(tf.rank(twoD)))

session.close()
