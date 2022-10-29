import tensorflow as tf

x = tf.Variable([1, 2, 3])
c = tf.constant([3, 4, 5])

print(x + c)

import tensorflow as tf

x = tf.constant(3.0)
with tf.GradientTape() as g:
    g.watch(x)
    y = x * x
# Gradient of y w.r.t. x
dy_dx = g.gradient(y, x)
print(dy_dx)


import tensorflow as tf

# Higher order derivatives too!
x = tf.constant(5.0)
with tf.GradientTape() as g:
    g.watch(x)
    with tf.GradientTape() as gg:
        gg.watch(x)
        y = x * x
    dy_dx = gg.gradient(y, x)  # dy_dx = 2 * x
d2y_dx2 = g.gradient(dy_dx, x)  # d2y_dx2 = 2
print(dy_dx)

print(d2y_dx2)
