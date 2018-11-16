# coding:utf-8
# 导入要用到的基本模块。
from __future__ import print_function
import numpy as np
import tensorflow as tf

# 创建图和Session
graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)

# tensorflow_inception_graph.pb文件中，既存储了inception的网络结构也存储了对应的数据
# 使用下面的语句将之导入
model_fn = 'tensorflow_inception_graph.pb'
#tf.gfile.FastGFile(path,decodestyle)
#功能：实现对图片的读取。
#参数：(1)path：图片所在路径 (2)decodestyle:图片的解码方式。(‘r’:UTF-8编码; ‘rb’:非UTF-8编码)
with tf.gfile.FastGFile(model_fn, 'rb') as f:
    #先创建一个空的图
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
# 定义t_input为我们输入的图像
t_input = tf.placeholder(np.float32, name='input')
imagenet_mean = 117.0
# 输入图像需要经过处理才能送入网络中
# expand_dims是加一维，从[height, width, channel]变成[1, height, width, channel]
# t_input - imagenet_mean是减去一个均值
t_preprocessed = tf.expand_dims(t_input - imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input': t_preprocessed})

# 找到所有卷积层
layers = [op.name for op in graph.get_operations() if op.type == 'Conv2D' and 'import/' in op.name]

# 输出卷积层层数
print('Number of layers', len(layers))
print('Layers:', layers)
# 特别地，输出mixed4d_3x3_bottleneck_pre_relu的形状
name = 'mixed4d_3x3_bottleneck_pre_relu'
print('shape of %s: %s' % (name, str(graph.get_tensor_by_name('import/' + name + ':0').get_shape())))
