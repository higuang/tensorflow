import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
tf.compat.v1.disable_eager_execution()
rand=tf.random.normal([4,6],1,0.9)
print(rand)


a=tf.Variable(initial_value=20.0)
b=tf.Variable(initial_value=40.0)
c = tf.add(a,b)
init=tf.compat.v1.global_variables_initializer();


with tf.compat.v1.Session() as sess:
    sess.run(init)
    print(c)