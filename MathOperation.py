import tensorflow as tf

x = tf.constant([100, 200, 300], name='x')
y = tf.constant([10, 4, 3], name='y')

#operation to add all elements in a 1 dimension tensor
sum_x = tf.reduce_sum(x, name='sum_x')

#operation to multiply all elements in a 1 dimension tensor
prod_y = tf.reduce_prod(y, name='prod_y')

#operation to divide two tensors
final_div = tf.div(sum_x, prod_y, name='final_div')

#operation to find the mean
final_mean = tf.reduce_mean([sum_x, prod_y], name='final_mean')

session = tf.Session()
print("Tensor x: ",session.run(x))
print("Tensor y: ",session.run(y))
print("Sum of tensor x: ",session.run(sum_x))
print("Product of tensor y: ", session.run(prod_y))
print("sum(x)/prod(y): ", session.run(final_div))
print("(sum(x) + (prod(y) ) / 2: ", session.run(final_mean))

session.close()