import tensorflow as tf

# y = Wx + b
W = tf.constant([10, 100], name='W')

# these placeholder can hold tensors of any shape
x = tf.placeholder(tf.int32, name='x')
b = tf.placeholder(tf.int32, name='b')

Wx = tf.multiply(W, x, name='Wx')
y = tf.add(Wx, b, name='y')

# y_ = x-b
y_ = tf.subtract(x, b, name='y_')

with tf.Session() as session:
    print("Intermediate result: Wx = ", session.run(Wx, feed_dict={x: [3, 33]}))

    print("Final result: Wx + b = ", session.run(y, feed_dict={x: [5, 50], y: [7, 9]}))

    print("Intermediate specifed: Wx + b = ", session.run(fetches=y, feed_dict={Wx : [100, 1000], b : [7, 9]}))

    print("Two results: [Wx+b, x- b] = ", session.run(fetches=[y, y_], feed_dict={y: [5, 50], y_: [7, 9]}))

writer = tf.summary.FileWriter('./fetch_example', session.graph)

writer.close()
