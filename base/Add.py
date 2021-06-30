import tensorflow as tf
import os
tf.compat.v1.disable_eager_execution()
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
a=tf.constant(1.0)
b=tf.constant(2.0)
c=tf.add(a,b)

#with 写法可以不用考虑关闭session
#allow_soft_placement=True,log_device_placement=True,可以把运行设备都打印出来了

with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True,log_device_placement=True)) as sess:
    print(sess.run(c))

#缓存可以用命令 tensorboard --logdir="D:\git_upload\github\tensorflow\base\data" 在浏览器上查看图
# tf.compat.v1.summary.FileWriter(".\data",graph=sess.graph)