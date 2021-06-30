import tensorflow as tf;
import os
#指定命名空间，在tensorboard结构清析
#样本,线性公式y=wx+b
with tf.compat.v1.variable_scope("o_data"):
    x=tf.random.normal([100,1],mean=0.0,stddev=1.0)
    y=tf.matmul(x,[[0.8]])+[[0.7]]


#建立模型trainable=False表示此参数不需要被训练
with tf.compat.v1.variable_scope("o_model"):
    w=tf.Variable(initial_value=tf.random.normal([1,1]),trainable=True,name="W")
    b=tf.Variable(initial_value=tf.random.normal([1,1]),name="B")
#y_pre 预测值,
    y_pre=tf.matmul(x,w)+b
#这里用均方误差损失函数(y-y_pre)^2/m
with tf.compat.v1.variable_scope("o_loss"):
    loss=tf.reduce_mean(tf.square(y-y_pre));

#梯度下降
with tf.compat.v1.variable_scope("o_optimizer"):
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
#收集变量在tensorboard显示
tf.summary.scalar("loss",loss)
tf.summary.histogram("weight",w)
tf.summary.histogram("isb",b)
#合并张量，固定写法
merge=tf.summary.merge_all();
#初始化
init=tf.compat.v1.global_variables_initializer()
with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True,log_device_placement=True)) as sess:
    sess.run(init);
    filewriter=tf.compat.v1.summary.FileWriter(".\data",graph=sess.graph)
    for i in range(500):
        sess.run(optimizer)
        #print("b",sess.run(b))
        #print("w",sess.run(w))
        #print("loss",sess.run(loss))
        summary=sess.run(merge)
        filewriter.add_summary(summary,i)


