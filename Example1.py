import tensorflow as tf

a = tf.constant(6, name='a')
b = tf.constant(3, name='b')
c = tf.constant(10, name='c')
d = tf.constant(5, name='d')

mul = tf.multiply(a, b, name='mul')
div = tf.div(c, d, name='div')
addn = tf.add_n([mul, div], name='addn')

session = tf.Session()
print(session.run(addn))

writer = tf.summary.FileWriter('./m2_example1', session.graph)
writer.close()
session.close()




