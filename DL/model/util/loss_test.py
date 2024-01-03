import tensorflow as tf
import numpy as np
from model.util.losses import *
import imageio
import cv2
def rearrange3d_fn(image):
    """ re-arrange image of shape[depth, height, width] into shape[height, width, depth]
    """
    image = np.squeeze(image)  # remove channels dimension
    # print('reshape : ' + str(image.shape))
    depth, height, width = image.shape
    image_re = np.zeros([height, width, depth])
    for d in range(depth):
        image_re[:, :, d] = image[d, :, :]
    return image_re
def rearrange3d_fn_inverse(image):
    """ re-arrange image of shape[depth, height, width] into shape[height, width, depth]
    """
    image = np.squeeze(image)  # remove channels dimension
    # print('reshape : ' + str(image.shape))
    height, width,depth = image.shape
    image_re = np.zeros([depth,height, width])
    for d in range(depth):
        image_re[d, :, :]=image[:, :, d]
    return image_re

def min_max(x,eps=1e-7):
    max_ = np.max(x)
    min_ =np.min(x)
    return (x-min_)/(max_-min_+eps)



def get_laplace_pyr(img,layer_num=3):
    batch,height,width,channel=img.shape
    lap_batch=[]
    def _gaussian(ori_image, down_times=3):
        temp_gau = ori_image.copy()
        gaussian_pyramid = [temp_gau]
        for i in range(down_times):

            temp_gau = cv2.pyrDown(temp_gau)
            gaussian_pyramid.append(temp_gau)
        return gaussian_pyramid
    def _laplacian(gaussian_pyramid, up_times=3):
        laplacian_pyramid = [gaussian_pyramid[-1]]
        for i in range(up_times, 0, -1):
            # i的取值为5,4,3,2,1,0也就是拉普拉斯金字塔有6层
            temp_pyrUp = cv2.pyrUp(gaussian_pyramid[i])
            temp_lap = cv2.subtract(gaussian_pyramid[i - 1], temp_pyrUp)
            laplacian_pyramid.append(temp_lap)
        return laplacian_pyramid
    for i in range(batch):
        gp = _gaussian(img[i],layer_num-1)
        lp = _laplacian(gp, layer_num - 1)
        lap_batch.append(lp)
    return lap_batch


def get_feed_dict(img_val,img_plh):
    assert len(img_val)==img_plh[0].get_shape().as_list()[0],'batch must be the same'
    assert len(img_val[0]) == len(img_plh),'layer num must be the same'

    feed_dict={}
    layers_=[]
    for idx in range(len(img_plh)):
        temp_batch=np.concatenate([img[idx] for img in img_val],axis=0)
        if temp_batch.ndim==3:
            temp_batch=temp_batch[np.newaxis,...]
        layers_.append(temp_batch)


    # layer1 = np.concatenate([img[0] for img in img_val],axis=0)
    # layer2 = np.concatenate([img[1] for img in img_val],axis=0)
    # layer3 = np.concatenate([img[2] for img in img_val],axis=0)

    for img_pyr,img_tensor in zip(layers_,img_plh):
         feed_dict.update({img_tensor:img_pyr})
    return feed_dict

batch_num=1
loss_func=l2_loss
ratio_list=[0.5,0.3,0.2]
image=tf.placeholder(tf.float32,[batch_num,220,220,220],'pred')
reference=tf.placeholder(tf.float32,[batch_num,220,220,220],'target')

mse_whole=loss_func(image,reference)
total_loss=0
layer_num=len(ratio_list)

base_shape=image.shape[1]//2**(layer_num-1)


batch, height, width, channel = image.get_shape().as_list()
img_py = []
ref_py = []
loss_list=[]

for i in range(layer_num ):
    temp_img = tf.placeholder(tf.float32, [batch, base_shape*2**(i), base_shape*2**(i), channel], 'pyr_img_%d'%i)
    temp_ref = tf.placeholder(tf.float32, [batch, base_shape*2**(i), base_shape*2**(i), channel], 'pyr_gt_%d' % i)
    temp_loss=l2_loss(temp_img,temp_ref)
    loss_list.append(temp_loss)
    total_loss+=ratio_list[i]*temp_loss
    img_py.append(temp_img)
    ref_py.append(temp_ref)
pass

lr_batch=min_max(rearrange3d_fn(imageio.volread(r'C:\Users\13774\Desktop\Slice\test\lr.tif')))
lr_batch=lr_batch[np.newaxis,...]
hr_batch=min_max(rearrange3d_fn(imageio.volread(r'C:\Users\13774\Desktop\Slice\test\hr.tif')))
hr_batch=hr_batch[np.newaxis,...]

lap_lr=get_laplace_pyr(img=lr_batch,layer_num=3)
lap_hr=get_laplace_pyr(img=hr_batch,layer_num=3)
pass


for img in lap_lr:
    for idx,lap_ in enumerate(img):
        temp=rearrange3d_fn_inverse(lap_).astype(np.float32, casting='unsafe')
        imageio.volwrite('lap_%i.tif'%idx,temp)


feed_dict1=get_feed_dict(lap_lr,img_py)
feed_dict1.update(get_feed_dict(lap_hr,ref_py))


with tf.Session() as sess:
    print('multi_scale_loss:%f'%(sess.run(total_loss,feed_dict1)))
    print('mse:%f'%(sess.run(mse_whole, {image:lr_batch,reference:hr_batch})))