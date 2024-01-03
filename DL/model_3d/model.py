import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import *
from model_3d.custom import SubpixelConv3d
w_init = tf.random_normal_initializer(stddev=0.02)
b_init = tf.constant_initializer(value=0.0)
g_init = tf.random_normal_initializer(1., 0.02)

from config_trans import config
device = config.TRAIN.device

def conv3d_transpose(input, out_channels, filter_size, stride, act=None, padding='SAME', name='conv3d_transpose' ):
    batch, depth, height, width, in_channels = input.outputs.get_shape().as_list()
    shape = [filter_size, filter_size, filter_size, out_channels, in_channels]
    output_shape = [batch, depth*stride, height*stride, width*stride, out_channels]
    strides = [1, stride, stride, stride, 1]
    return tl.layers.DeConv3dLayer(input, act=act, shape=shape, output_shape=output_shape, strides=strides, padding=padding, W_init=w_init, b_init=b_init, name=name);
def conv3d_transpose2(input, out_channels, filter_size, stride, padding='SAME', name='conv3d_transpose' ):
    return tf.layers.Conv3DTranspose(filters=out_channels, kernel_size=(filter_size, filter_size, filter_size), strides=(stride, stride, stride), padding=padding, kernel_initializer=w_init, bias_initializer=b_init, name=name);

def conv3d(layer, out_channels, filter_size=3, stride=1, act=tf.identity, padding='SAME', W_init=w_init, b_init=b_init, name='conv3d'):
    """
    Params
        shape - shape of the filters, [filter_depth, filter_height, filter_width, in_channels, out_channels].
        strides - shift in [batch in_depth in_height in_width in_channels] 
    """
    in_channels = layer.outputs.get_shape().as_list()[-1]
    shape=[filter_size,filter_size,filter_size,in_channels,out_channels]
    strides=[1,stride,stride,stride,1]
    return Conv3dLayer(layer, act=act, shape=shape, strides=strides, padding=padding, W_init=W_init, b_init=b_init, name=name)
    
def conv3d2(input, out_channels, filter_size=3, stride=1, padding='SAME', name='conv3d'):
    in_channels = input.get_shape().as_list()[-1]
    filter_shape = [filter_size, filter_size, filter_size, in_channels, out_channels]
    strides = [1, stride, stride, stride, 1]
   
    with tf.variable_scope(name):
        weight = tf.get_variable(name='weight_conv3d', shape=filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.02))
        bias = tf.get_variable(name='bias_conv3d', shape=[out_channels], initializer=tf.constant_initializer(value=0.0))
        return tf.nn.conv3d(input, weight, strides, padding)
        
def concat(layers, concat_dim=4, name='concat'):
    return tl.layers.ConcatLayer(layers, concat_dim=concat_dim, name=name)

def batch_norm(layer, act=tf.identity, is_train=True, gamma_init=g_init, name='bn'):  
    return BatchNormLayer(layer, act=act, is_train=is_train, gamma_init=gamma_init, name=name)
    
def deep_feature_extractor(input, reuse=False, name="dfe"):
    n_layers_encoding = 3;
    n_channels = 64
    n_channels_in = input.get_shape().as_list()[-1];
    features = []
    encoding = []
    with tf.device('/device:GPU:%d' %  device):
        with tf.variable_scope(name, reuse=reuse):
            n = InputLayer(input)
            for i in range(1, n_layers_encoding + 1):
                n = conv3d(n, out_channels=n_channels*i, filter_size=3, stride=2, act=tf.nn.relu, padding='SAME', name='conv%d' % i)
                encoding.append(n)
                features.append(n.outputs)
            
            
            for i in range(1, n_layers_encoding + 1):
                c = n_channels*(n_layers_encoding - i)
                if c == 0:
                    c = 32
                n = conv3d_transpose(n, out_channels=c, filter_size=3, stride=2, act=tf.nn.relu, padding='SAME', name='conv3d_transpose%d' % i )
                if n_layers_encoding - i > 0:
                    n = concat([encoding[n_layers_encoding - i - 1], n], name='concat%d' % i)
            n = conv3d(n, out_channels=n_channels_in, filter_size=3, stride=1, act=tf.tanh, padding='SAME', name='out')
            
            return n, features

def deep_feature_loss(features_img, features_ref):
    assert len(features_img) == len(features_ref)
    diff = []   
    for i in range(0, len(features_img)):
        f_img = features_img[i] 
        f_ref = features_ref[i]
        f_diff = mean_squared_error(f_img, f_ref)
        diff.append(f_diff)
    diff_array = np.asarray(diff)
    diff_mean = diff_array.sum() / len(features_ref)
    return diff_mean
    
def bottleneck(layer, is_train=True, name='bottleneck'):
    nchannels = layer.outputs.get_shape().as_list()[-1]
    with tf.variable_scope(name):
        n = conv3d(layer, shape=[1, 1, 1, nchannels, 16], name='reduce')
        n = batch_norm(n, is_train=is_train, name='reduce/bn')
        n = conv3d(n, shape=[3,3,3,16,16], name='mid')
        n = batch_norm(n, is_train=is_train, name='mid/bn')
        n = conv3d(n, shape=[1,1,1,16,nchannels], name='expand')
        return n
        
def res_block(layer, is_train=True, name='block'):
    with tf.variable_scope(name):
        #n = conv3d(layer, shape=[3,3,3,64,64], name='conv3d1')
        n = bottleneck(layer, is_train=is_train, name='bottleneck1')
        n = batch_norm(n, act=tf.nn.relu, is_train=is_train, name='bn1')
        #n = conv3d(n, shape=[3,3,3,64,64], name='conv3d2')
        n = bottleneck(n, is_train=is_train, name='bottleneck2')
        n = batch_norm(n, act=tf.nn.relu, is_train=is_train,  name='bn2')
        n = ElementwiseLayer([layer, n], combine_fn=tf.add)
        return n

def res_blocks(n, n_blocks=8, out_channels=64, is_train=False, name='res_blocks'):
    with tf.variable_scope(name):
        temp = n
        # Residual Blocks
        for i in range(n_blocks):
            n = res_block(n, is_train=is_train, name='block%d' % i) 
            
        n = conv3d(n, shape=[3,3,3,64,out_channels], name='conv3d2')
        n = batch_norm(n, is_train=is_train, name='bn2')
        n = ElementwiseLayer([n, temp], combine_fn=tf.add)
        return n    
            
def generator3d(lr, scale=4, is_train=False, reuse=False):
    #assert scale == 1 or scale == 2 or scale == 4
    
    with tf.device('/device:GPU:0'):
        with tf.variable_scope("generator3d", reuse=reuse) as vs:
            tl.layers.set_name_reuse(reuse)
            n = InputLayer(lr, name='in') # n must have a shape of [batch, in_depth, in_height, in_width, in_channels] for next Conv3dLayer
            n = conv3d(n, act=tf.nn.relu, shape=[3,3,3,1,64], name='conv3d1')
            
            n = SubpixelConv3d(n, scale=2, name='subpixel1')
            n = conv3d(n, act=tf.nn.relu, shape=[3,3,3,8,64], name='subpixel1/conv3d')
            
            n = res_blocks(n, n_blocks=3, out_channels=64, is_train=is_train, name='res1')
            
            n = SubpixelConv3d(n, scale=2, name='subpixel2')
            n = conv3d(n, act=tf.nn.relu, shape=[3,3,3,8,64], name='subpixel2/conv3d')
            
            n = res_blocks(n, n_blocks=3, out_channels=64, is_train=is_train, name='res2')
            
            '''
            if scale == 1:
                
                n = Conv3dLayer(n, act=tf.nn.tanh, shape=[1,1,1,64,4], strides=[1,1,1,1,1], padding='SAME', W_init=w_init, name='n4s1/1')
                
            else :       
                n = Conv3dLayer(n, act=tf.identity, shape=[3,3,3,64,128], strides=[1,1,1,1,1], padding='SAME', W_init=w_init, b_init=b_init, name='n128s1/1')        
                n = SubpixelConv3d(n, scale=2, n_out_channel=16, name='subpixel1')
            
                if scale == 2:
                    n = Conv3dLayer(n, act=tf.nn.tanh, shape=[1,1,1,16,4], strides=[1,1,1,1,1], padding='SAME', W_init=w_init, name='n4s1/2')
                
                elif scale == 4:
                    n = Conv3dLayer(n, act=tf.identity, shape=(3,3,3,16,32), strides=[1,1,1,1,1], padding='SAME', W_init=w_init, b_init=b_init, name='n32s1/2')
                    n = SubpixelConv3d(n, scale=2, n_out_channel=4, name='subpixel2')
                else:
                    raise Exception("undefined scale : %d " % scale);
            '''
            n = conv3d(n, act=tf.tanh, shape=[1,1,1,64,1], strides=[1,1,1,1,1], name='out')
            return n  

def discriminator3d(input_imaegs, is_train=True, reuse=False):
    
    df_dim = 64
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    
    with tf.device('/device:GPU:1'):
        with tf.variable_scope("discriminator3d", reuse=reuse):
            tl.layers.set_name_reuse(reuse)
            net_in = InputLayer(input_imaegs, name='in')
            net_ho = conv3d(net_in, act=lrelu, shape=[4,4,4,1,df_dim], strides=[1,2,2,2,1], name='h0/c')
            
            net_h1 = conv3d(net_ho, shape=[4,4,4,df_dim, df_dim*2], strides=[1,2,2,2,1], name='h1/c')
            net_h1 = batch_norm(net_h1, act=lrelu, is_train=is_train, name='h1/bn')
            
            net_h2 = conv3d(net_h1, shape=[4,4,4,df_dim*2, df_dim*4], strides=[1,2,2,2,1], name='h2/c')
            net_h2 = batch_norm(net_h2, act=lrelu, is_train=is_train, name='h2/bn')
        
            net_h3 = conv3d(net_h2, shape=[4,4,4,df_dim*4, df_dim*8], strides=[1,2,2,2,1], name='h3/c')
            net_h3 = batch_norm(net_h3, act=lrelu, is_train=is_train, name='h3/bn')
            
            net_h4 = conv3d(net_h3, shape=[4,4,4,df_dim*8, df_dim*16], strides=[1,2,2,2,1], name='h4/c')
            net_h4 = batch_norm(net_h4, act=lrelu, is_train=is_train, name='h4/bn')

            net_h5 = conv3d(net_h4, shape=[4,4,4,df_dim*16, df_dim*32], strides=[1,2,2,2,1], name='h5/c')
            net_h5 = batch_norm(net_h5, act=lrelu, is_train=is_train, name='h5/bn')
            
            net_h6 = conv3d(net_h5, act=tf.identity, shape=[1,1,1,df_dim*32, df_dim*16], strides=[1,1,1,1,1], padding='SAME', W_init=w_init, b_init=b_init, name='h6/c')
            net_h6 = batch_norm(net_h6, act=lrelu, is_train=is_train, name='h6/bn')
            
            net_h7 = conv3d(net_h6, act=tf.identity, shape=[1,1,1,df_dim*16, df_dim*8], strides=[1,1,1,1,1], padding='SAME', W_init=w_init, b_init=b_init, name='h7/c')
            net_h7 = batch_norm(net_h7, act=lrelu, is_train=is_train, name='h7/bn')
            
            net = conv3d(net_h7, act=tf.identity, shape=[1,1,1,df_dim*8, df_dim*2], strides=[1,1,1,1,1], padding='SAME', W_init=w_init, b_init=b_init, name='res/c1')
            net = batch_norm(net, act=lrelu, is_train=is_train, name='res/bn1')
            
            net = conv3d(net, act=tf.identity, shape=[3,3,3,df_dim*2, df_dim*2], strides=[1,1,1,1,1], padding='SAME', W_init=w_init, b_init=b_init, name='res/c2')
            net = batch_norm(net, act=lrelu, is_train=is_train, name='res/bn2')
            
            net = conv3d(net, act=tf.identity, shape=[3,3,3,df_dim*2, df_dim*8], strides=[1,1,1,1,1], padding='SAME', W_init=w_init, b_init=b_init, name='res/c3')
            net = batch_norm(net, is_train=is_train, name='res/bn3')
            
            net_h8 = ElementwiseLayer(layer=[net_h7, net], combine_fn=tf.add, name='res/add')
            net_h8.outputs = tl.act.lrelu(net_h8.outputs, 0.2)
            
            net_h0 = FlattenLayer(net_h8, name='h0/flatten')
            net_h0 = DenseLayer(net_h0, n_units=1, act=tf.identity, W_init=w_init, name='h0/dense')
            logits = net_h0.outputs
            net_h0.outputs = tf.nn.sigmoid(net_h0.outputs)
            
            return net_h0, logits
        
    
def mean_squared_error(output, target, is_mean=False, name="mean_squared_error"):
    """ Return the TensorFlow expression of mean-square-error (L2) of two batch of data.

    Parameters
    ----------
    output : 2D, 3D or 4D tensor i.e. [batch_size, n_feature], [batch_size, w, h] or [batch_size, w, h, c].
    target : 2D, 3D or 4D tensor.
    is_mean : boolean, if True, use ``tf.reduce_mean`` to compute the loss of one data, otherwise, use ``tf.reduce_sum`` (default).

    References
    ------------
    - `Wiki Mean Squared Error <https://en.wikipedia.org/wiki/Mean_squared_error>`_
    """
    with tf.name_scope(name):
        if output.get_shape().ndims == 2:   # [batch_size, n_feature]
            if is_mean:
                mse = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(output, target), 1))
            else:
                mse = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(output, target), 1))
        elif output.get_shape().ndims == 3: # [batch_size, w, h]
            if is_mean:
                mse = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(output, target), [1, 2]))
            else:
                mse = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(output, target), [1, 2]))
        elif output.get_shape().ndims == 4: # [batch_size, w, h, c]
            if is_mean:
                mse = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(output, target), [1, 2, 3]))
            else:
                mse = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(output, target), [1, 2, 3]))
                
        elif output.get_shape().ndims == 5: # [batch_size, depth, height, width, channels]
            if is_mean:
                mse = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(output, target), [1, 2, 3, 4]))
            else:
                mse = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(output, target), [1, 2, 3, 4]))
        else:
            raise Exception("Unknow dimension")
        return mse
        
"""   
def cross_entropy_(x, y):
    NEAR_0 = 1e-10
    x_ = x + 1. / 2.   #normalized [0,1]
    y_ = y + 1. / 2.
    cross_entropy_ = -(y_*tf.log(x_+NEAR_0)+(1-y_)*tf.log(1-x_+NEAR_0))
    return cross_entropy_
"""    

def cross_entropy_(output, target): 
    batch_size = config.TRAIN.batch_size
    logits_real = tf.reshape(output, (batch_size, -1))
    labels_real = tf.reshape(target, (batch_size, -1))
    #labels_real = labels_real + 1. / 2 
    labels_real = labels_real + 1.  #[0, 2]
    labels_real = labels_real / 2.
    cross_entropy_ = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_real, logits=logits_real)
    #print(cross_entropy_.shape)
    return cross_entropy_
 
def cross_entropy(output, target, is_mean=False, name="cross_entropy"):
    with tf.name_scope(name):
        if output.get_shape().ndims == 2:   # [batch_size, n_feature]
            if is_mean:
                cross_entropy = tf.reduce_mean(tf.reduce_mean(cross_entropy_(output, target), 1))
            else:
                cross_entropy = tf.reduce_mean(tf.reduce_sum(cross_entropy_(output, target), 1))
        elif output.get_shape().ndims == 3: # [batch_size, w, h]
            if is_mean:
                cross_entropy = tf.reduce_mean(tf.reduce_mean(cross_entropy_(output, target), [1, 2]))
            else:
                cross_entropy = tf.reduce_mean(tf.reduce_sum(cross_entropy_(output, target), [1, 2]))
        elif output.get_shape().ndims == 4: # [batch_size, w, h, c]
            if is_mean:
                cross_entropy = tf.reduce_mean(tf.reduce_mean(cross_entropy_(output, target), [1, 2, 3]))
            else:
                cross_entropy = tf.reduce_mean(tf.reduce_sum(cross_entropy_(output, target), [1, 2, 3]))
                
        elif output.get_shape().ndims == 5: # [batch_size, depth, height, width, channels]
            if is_mean:
                cross_entropy = tf.reduce_mean(tf.reduce_mean(cross_entropy_(output, target), axis=1))
            else:
                cross_entropy = tf.reduce_mean(tf.reduce_mean(cross_entropy_(output, target), axis=1))
        else:
            raise Exception("Unknow dimension")
        return cross_entropy

def l2_loss(image, reference):
  with tf.variable_scope('l2_loss'):
    return tf.reduce_mean(tf.squared_difference(image, reference))
        
def sobel_edges(input):
    '''
    find the edges of the input image, using the bulit-in tf function

    Params: 
        -input : tensor of shape [batch, depth, height, width, channels]
    return:
        -tensor of the edges: [batch, height, width, depth]
    '''
    # transpose the image shape into [batch, h, w, d] to meet the requirement of tf.image.sobel_edges
    img = tf.squeeze(tf.transpose(input, perm=[0,2,3,1,4]), axis=-1) 
    
    # the last dim holds the dx and dy results respectively
    edges_xy = tf.image.sobel_edges(img)
    #edges = tf.sqrt(tf.reduce_sum(tf.square(edges_xy), axis=-1))

    return edges_xy

def edges_loss(image, reference):
    '''
    params: 
        -image : tensor of shape [batch, depth, height, width, channels], the output of DVSR
        -reference : same shape as the image
    '''
    with tf.variable_scope('edges_loss'):
        edges_sr = sobel_edges(image)
        edges_hr = sobel_edges(reference)
        
        #return tf.reduce_mean(tf.abs(edges_sr - edges_hr))
        return l2_loss(edges_sr, edges_hr)



def tf_ssim(img1, img2, cs_map=False, mean_metric=True, filter_size=11, filter_sigma=1.5):
    _, depth, height, width, ch = img1.get_shape().as_list()
    size = min(filter_size, height, width, depth)
    sigma = size * filter_sigma / filter_size if filter_size else 0

    window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2

    padded_img1 = tf.pad(img1, [[0, 0], [size//2, size//2], [size//2, size//2], [size//2, size//2], [0, 0]], mode="CONSTANT")
    padded_img2 = tf.pad(img2, [[0, 0], [size//2, size//2], [size//2, size//2], [size//2, size//2], [0, 0]], mode="CONSTANT")
    mu1 = tf.nn.conv3d(padded_img1, window, strides=[1,1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv3d(padded_img2, window, strides=[1,1,1,1,1], padding='VALID')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2

    paddedimg11 = padded_img1*padded_img1
    paddedimg22 = padded_img2*padded_img2
    paddedimg12 = padded_img1*padded_img2

    sigma1_sq = tf.nn.conv3d(paddedimg11, window, strides=[1,1,1,1,1], padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv3d(paddedimg22, window, strides=[1,1,1,1,1], padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv3d(paddedimg12, window, strides=[1,1,1,1,1], padding='VALID') - mu1_mu2
    ssim_value = tf.clip_by_value(((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2)), 0, 1)
    if cs_map:
        cs_map_value = tf.clip_by_value((2*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2), 0, 1)
        value = (ssim_value, cs_map_value)
    else:
        value = ssim_value
    if mean_metric:
        value = tf.reduce_mean(value)
    return value


def tf_ms_ssim_resize(img1, img2, weights=None, return_ssim_map=None, filter_size=11, filter_sigma=1.5):
    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    level = len(weights)
    assert return_ssim_map is None or return_ssim_map < level
    weight = tf.constant(weights, dtype=tf.float32)
    mssim = []
    mcs = []
    _, h, w, _ = img1.get_shape().as_list()
    for l in range(level):
        ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False, filter_size=filter_size, filter_sigma=filter_sigma)
        if return_ssim_map == l:
            return_ssim_map = tf.image.resize_images(ssim_map, size=(h, w), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        mssim.append(tf.reduce_mean(ssim_map))
        mcs.append(tf.reduce_mean(cs_map))
        img1 = tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='SAME')
        img2 = tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='SAME')

    # list to tensor of dim D+1
    mssim = tf.stack(mssim, axis=0)
    mcs = tf.stack(mcs, axis=0)

    value = tf.reduce_prod(mcs[0:level-1]**weight[0:level-1])*(mssim[level-1]**weight[level-1])
    if return_ssim_map is not None:
        return value, return_ssim_map
    else:
        return value


def tf_ms_ssim(img1, img2, weights=None, mean_metric=False):
    if weights is None:
        weights = [1, 1, 1, 1, 1] # [0.0448, 0.2856, 0.3001, 0.2363, 0.1333] #[1, 1, 1, 1, 1] #
    level = len(weights)
    sigmas = [0.5]
    for i in range(level-1):
        sigmas.append(sigmas[-1]*2)
    weight = tf.constant(weights, dtype=tf.float32)
    mssim = []
    mcs = []
    for l, sigma in enumerate(sigmas):
        filter_size = int(max(sigma*4+1, 11))
        ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False, filter_size=filter_size, filter_sigma=sigma)
        mssim.append(ssim_map)
        mcs.append(cs_map)
    # list to tensor of dim D+1
    value = mssim[level-1]**weight[level-1]
    for l in range(level):
        value = value * (mcs[l]**weight[l])
    if mean_metric:
        return tf.reduce_mean(value)
    else:
        return value