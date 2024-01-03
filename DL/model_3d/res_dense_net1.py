import tensorflow as tf
import tensorlayer as tl
from model_3d.model import conv3d, SubpixelConv3d, conv3d_transpose
from tensorlayer.layers import InputLayer, ConcatLayer, ElementwiseLayer

conv_kernel = 3

def concat(layer, concat_dim=-1, name='concat'):
    return ConcatLayer(layer, concat_dim=concat_dim, name=name)
   
def res_dense_block(preceding, G=64, name='rdb'):
    """
    Resifual dense block
    Params : 
        preceding - An Layer class, feature maps of preceding block 
        G         - Growth rate of feature maps
    """
    G0 = preceding.outputs.shape[-1]
    if G0 != G:
        raise Exception('G0 and G must be equal in RDB')
        
    with tf.variable_scope(name):
        n1 = conv3d(preceding, out_channels=G, filter_size=conv_kernel, stride=1, act=tf.nn.relu, name='conv1')
        n2 = concat([preceding, n1], name='conv2_in')
        n2 = conv3d(n2, out_channels=G, filter_size=conv_kernel, stride=1, act=tf.nn.relu, name='conv2')
        
        n3 = concat([preceding, n1, n2], name='conv3_in')
        n3 = conv3d(n3, out_channels=G, filter_size=conv_kernel, stride=1, act=tf.nn.relu, name='conv3')
        
        # local feature fusion (LFF)
        n4 = concat([preceding, n1, n2, n3], name='conv4_in')
        n4 = conv3d(n4, out_channels=G, filter_size=1, name='conv4')
        
        # local residual learning (LRL)
        out = ElementwiseLayer([preceding, n4], combine_fn=tf.add, name='out')
        
        return out

def upscale(layer, scale=2, name='upscale'):
    return SubpixelConv3d(layer, scale=scale, n_out_channel=None, act=tf.identity, name=name)
        
def res_dense_net(lr, reuse=False, name='generator'):
    G0 = 64
    with tf.variable_scope(name, reuse=reuse):
        with tf.variable_scope('RDN'):
            n = InputLayer(lr, 'lr')
            
            # shallow feature extraction layers
            n1 = conv3d(n, out_channels=G0, filter_size=conv_kernel, name='shallow1')
            n2 = conv3d(n1, out_channels=G0, filter_size=conv_kernel, name='shallow2')
            
            n3 = res_dense_block(n2, name='rdb1')
            n4 = res_dense_block(n3, name='rdb2')
            n5 = res_dense_block(n4, name='rdb3')

            # global feature fusion (GFF)
            n6 = concat([n3, n4, n5], name='gff')
            n6 = conv3d(n6, out_channels=G0, filter_size=1, name='gff/conv1')
            n6 = conv3d(n6, out_channels=G0, filter_size=conv_kernel, name='gff/conv2')
            
            # global residual learning 
            n7 = ElementwiseLayer([n6, n1], combine_fn=tf.add, name='grl')
            
            #n8 = upscale(n7, scale=2, name='upscale1')
            #n9 = upscale(n8, scale=2, name='upscale2')
            #out = conv3d(n9, out_channels=1, filter_size=conv_kernel, act=tf.tanh, name='out')
            
            out = conv3d(n7, out_channels=1, filter_size=conv_kernel, act=tf.tanh, name='out')
            
            return out
            """
            n8_out = conv3d(n7, out_channels=1, filter_size=conv_kernel, name='logits')
            logits = n8_out.outputs
            n8_out.outputs = tf.nn.tanh(n8_out.outputs, name='out')
            
            return n8_out, logits
            """
            
            
        
        