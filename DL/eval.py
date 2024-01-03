import imageio

from utils import *
from config_test import config
label = config['label']
n_num = config['img_setting'].Nnum
n_slices = config['img_setting'].n_slices
sr_factor = config['img_setting'].sr_factor
normalize_mode = config['preprocess'].normalize_mode
if normalize_mode == 'normalize_percentile_z_score':
    normalize_fn = normalize_percentile_z_score
elif normalize_mode == 'percentile':
    normalize_fn = normalize_percentile
elif normalize_mode == 'constant':
    normalize_fn = normalize_constant
else:
    normalize_fn = normalize




def __raise(info):
    raise Exception(info)


def ViewMap2LFP(input, n_num):
    batch, h, w, channel = input.shape
    base_h = h // n_num
    base_w = w // n_num
    assert channel == 1, 'wrong channel of net prediction'
    LFP_ = np.zeros(input.shape, np.float32)
    for i in range(batch):
        for v in range(n_num):
            for u in range(n_num):
                LFP_[i, v::n_num, u::n_num, :] = input[:, v * base_h:(v + 1) * base_h, u * base_w:(u + 1) * base_w, :]
    return LFP_


def read_valid_images(path):
    """return images in shape [n_images, height=img_size/n_num, width=img_size/n_num, channels=n_num**2]
    """

    def __cast(im, dtype=np.float32):
        return im if im.dtype is np.float32 else im.astype(np.float32, casting='unsafe')

    img_list = sorted(tl.files.load_file_list(path=path, regx='\.tif', printable=False))
    img_set=[]
    for img_file in img_list:
        img_temp=get_2d_lf(img_file, path, n_num=n_num,normalize_fn=None)
        print('loading %s'%img_file)
        img_temp = normalize_fn(img_temp,gamma=config['img_gamma'])
        img_set.append(__cast(img_temp))

    # img_set = [__cast(get_2d_lf(img_file, path, n_num=n_num, normalize_fn=normalize_fn)) for img_file in img_list]
    len(img_set) != 0 or __raise("none of the images have been loaded")
    print('read %d from %s' % (len(img_set), path))
    img_set = np.asarray(img_set)
    _, height, width, _ = img_set.shape

    return img_set, img_list, height, width


def infer(epoch, batch_size=1, use_cpu=False):
    """ Infer the 3-D images from the 2-D LF images using the trained VCD-Net

    Params:
        -epoch     : int, the epoch number of the checkpoint file to be loaded
        -batch_size: int, batch size of the VCD-Net
        -use_cpu   : bool, whether to use cpu for inference. If false, gpu will be used.
    """
    val_epoch = config['eval_ckpt']
    if val_epoch=='best':
        epoch='best'
    else:
        epoch = str(val_epoch)
    # checkpoint_dir = config.Trans.ckpt_dir if args.trans else config.TRAIN.ckpt_dir
    checkpoint_dir = os.path.join(config['root_path'],'DL','checkpoint',label)
    valid_lr_img_path = config['validation_data_path'][0]
    save_dir = os.path.join(valid_lr_img_path,'Recon_%s'%label)
    tl.files.exists_or_mkdir(save_dir)

    # layers_save_path  = os.path.join(save_dir,'layers_outputs')
    # tl.files.exists_or_mkdir(layers_save_path)


    valid_lf_extras, names, height, width = read_valid_images(valid_lr_img_path)
    t_image = tf.placeholder('float32', [batch_size, height, width, 1])
    input_size = [height, width]
    SR_size = sr_factor * np.array([height, width])
    Recon_size = np.multiply(SR_size, config['img_setting'].ReScale_factor)
    device_str = '/gpu:0' if not use_cpu else '/cpu:0'

    ngf = config['net_setting'].ngf
    denoise_tag = config['net_setting'].denoise_model
    ngf1 = ngf[0]
    SR_tag = config['net_setting'].SR_model
    ngf2 = ngf[1]
    Recon_tag = config['net_setting'].Recon_model
    ngf3 = ngf[2]
    print('[!] Denoise:%s\n[!] SR:%s\n[!] Recon:%s' % (denoise_tag, SR_tag, Recon_tag))
    denoise_model = eval(denoise_tag)
    sr_model = eval(SR_tag)
    recon_model = eval(Recon_tag)

    with tf.device(device_str):

        denoise_net = denoise_model(LFP=t_image, output_size=input_size, sr_factor=None, angRes=n_num,
                                    reuse=False, name=denoise_tag, channels_interp=ngf1,
                                    normalize_mode=normalize_mode)
        SR_net = sr_model(LFP=denoise_net.outputs, output_size=SR_size, sr_factor=sr_factor,
                          angRes=n_num, reuse=False, name=SR_tag, channels_interp=ngf2,
                          normalize_mode=normalize_mode, transform_layer='SAI2Macron')
        Recon_net = recon_model(lf_extra=SR_net.outputs, n_slices=n_slices, output_size=Recon_size,
                                is_train=True, reuse=False, name=Recon_tag, channels_interp=ngf3,
                                normalize_mode=normalize_mode, transform='SAI2ViewStack',pyrimid_list=config['net_setting'].Unetpyrimid_list)


    # denoise_ckpt= [filename for filename in os.listdir(checkpoint_dir) if
    #                 ('.npz' in filename and epoch in filename and 'denoise' in filename)]
    denoise_ckpt= [filename for filename in os.listdir(checkpoint_dir) if
                    ('.npz' in filename and epoch in filename and 'denoise' in filename)]
    # SR_ckpt_file = [filename for filename in os.listdir(checkpoint_dir) if
    #                 ('.npz' in filename and '50' in filename and 'SR' in filename)]
    SR_ckpt_file = [filename for filename in os.listdir(checkpoint_dir) if
                    ('.npz' in filename and epoch in filename and 'SR' in filename)]
    Recon_ckpt_file = [filename for filename in os.listdir(checkpoint_dir) if
                       ('.npz' in filename and epoch in filename and 'recon' in filename)]

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        tl.layers.initialize_global_variables(sess)

        tl.files.load_and_assign_npz(sess=sess, name=os.path.join(checkpoint_dir, denoise_ckpt[0]),
                                     network=denoise_net)


        tl.files.load_and_assign_npz(sess=sess, name=os.path.join(checkpoint_dir, SR_ckpt_file[0]),
                                     network=SR_net)
        tl.files.load_and_assign_npz(sess=sess, name=os.path.join(checkpoint_dir, Recon_ckpt_file[0]),
                                     network=Recon_net)

        # #im_buffer        = np.zeros([len(valid_lf_extras), height * n_num, width * n_num, config.PSF.n_slices])
        # im_buffer        = []
        # recon_start_time = time.time()

        recon_time = 0
        # print('normlize_mode:%s net_tag:%s -- %s' % (normalize_mode, config.net_setting.SR_model, config.net_setting.Recon_model))

        for idx in range(0, len(valid_lf_extras), batch_size):

            start_time = time.time()
            # denoise_out = sess.run(denoise_net.outputs, {t_image: valid_lf_extras[idx:idx + batch_size]})
            # sr_out = sess.run(SR_net.outputs, {t_image: valid_lf_extras[idx:idx + batch_size]})
            recon_out = sess.run(Recon_net.outputs, {t_image: valid_lf_extras[idx:idx + batch_size]})

            batch_time = time.time() - start_time
            recon_time = recon_time + batch_time
            recon_out = np.clip(recon_out,a_min=0,a_max=1)

            # denoise_out=np.squeeze(denoise_out)
            # sr_out=np.squeeze(sr_out)
            recon_out=np.squeeze(recon_out)

            # imageio.imwrite(os.path.join(save_dir , '%s-%s' % (config['net_setting'].denoise_model, names[idx])),
            #                             denoise_out)
            # imageio.imwrite(os.path.join(save_dir , '%s-%s' % (config['net_setting'].SR_model, names[idx])),sr_out)
            imageio.volwrite(os.path.join(save_dir , '%s-%s' % (config['net_setting'].Recon_model, names[idx])),np.transpose(np.transpose(recon_out, [1, 2, 0]), [1, 2, 0]))
            # imageio.imwrite(os.path.join(save_dir, '%s-%s' % (config['net_setting'].Recon_model, names[idx])),
            #                  np.max(np.transpose(np.transpose(recon_out, [1, 2, 0]), [1, 2, 0]),axis=0))
            # save_tiff_imagej_compatible(os.path.join(save_dir , '%s-%s' % (config['net_setting'].denoise_model, names[idx])),
            #                             np.squeeze(denoise_out), axes='YX')
            # save_tiff_imagej_compatible(os.path.join(save_dir , '%s-%s' % (config['net_setting'].SR_model, names[idx])),
            #                             np.squeeze(sr_out), axes='YX')
            # save_tiff_imagej_compatible(os.path.join(save_dir , '%s-%s' % (config['net_setting'].Recon_model, names[idx])),
            #                             np.squeeze(recon_out), axes='YXZ')

            # recon_out = np.squeeze((recon_out-np.amin(recon_out))/(np.amax(recon_out)-np.amin(recon_out)))*255
            # start_time = time.time()
            # save_tiff_imagej_compatible(save_dir + '%s-%s' % (config.net_setting.Recon_model, names[idx]),recon_out.astype(np.uint8), axes='YXZ')
            print("\rtime elapsed (sess.run): %4.4fs " % (time.time() - start_time), end='')
if __name__ == '__main__':
    import argparse
    import os
    import time
    import numpy as np
    from model import LF_SA_small,LF_attention_denoise,MultiRes_UNet
    # from csbdeep.io import save_tiff_imagej_compatible

    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, default=0, help='')
    parser.add_argument('-c', '--ckpt', type=int, default=0)
    parser.add_argument('-b', '--batch', type=int, default=1)

    # parser.add_argument('--trans', type=bool, default=False, help='')
    parser.add_argument("--cpu", help="use CPU instead of GPU for inference",
                        action="store_true")
    args = parser.parse_args()

    ckpt = args.ckpt
    batch_size = args.batch
    use_cpu = args.cpu
    use_cpu = True if args.gpu==-1 else False
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import tensorflow as tf
    import tensorlayer as tl

    infer(ckpt, batch_size=batch_size, use_cpu=use_cpu)