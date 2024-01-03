import tensorflow as tf
from .custom import GlobalMeanPooling3d,GlobalMaxPooling3d,DenseLayer_reuse
from tensorlayer.layers import ElementwiseLayer,DenseLayer






def local_dense(layer,channel,ratio,reuse=False,name='local_dense'):
    kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
    bias_initializer = tf.constant_initializer(value=0.0)

    n=DenseLayer_reuse(prev_layer=layer,
                 n_units=channel // ratio,
                 act=tf.nn.relu,
                 W_init=kernel_initializer,
                 b_init=bias_initializer,
                 name='mlp_0',
                reuse=reuse
                )
    n=DenseLayer_reuse(prev_layer=n,
                 n_units=channel,
                 W_init=kernel_initializer,
                 b_init=bias_initializer,
                 name='mlp_1',
                reuse = reuse
                )
    return n

def channel_attention3d(layer,reduction_ratio=8,name='attention_layer'):

    with tf.variable_scope(name):
        channel = layer.outputs.get_shape()[-1]
        avg_pool= GlobalMeanPooling3d(layer,name='meanpooling')
        max_pool = GlobalMaxPooling3d(layer, name='maxpooling')
        avg_pool = local_dense(avg_pool,channel,reduction_ratio,reuse=False,name='local_dense_avg')
        max_pool = local_dense(max_pool, channel, reduction_ratio, reuse=True, name='local_dense_max')

        avg_pool.outputs = avg_pool.outputs[:,tf.newaxis,tf.newaxis,tf.newaxis,:]
        max_pool.outputs = max_pool.outputs[:, tf.newaxis, tf.newaxis, tf.newaxis, :]

        scale = ElementwiseLayer([avg_pool,max_pool],combine_fn=tf.add)
        scale.outputs = tf.sigmoid(scale.outputs)

        out_layer=ElementwiseLayer([layer,scale],combine_fn=tf.multiply)
    return out_layer
