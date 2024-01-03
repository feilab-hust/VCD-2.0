import numpy as np
import imageio
import PIL.Image as pilimg
import tensorlayer as tl
import  os
# from csbdeep.io import save_tiff_imagej_compatible
from skimage import io
# import cv2
import mat73

__all__ = [
    'get_img3d_fn',
    'rearrange3d_fn',
    'get_and_rearrange3d',
    'get_img2d_fn',
    'get_lf_extra',
    'get_2d_lf',
    'lf_extract_fn',
    'write3d',
    # 'write3d_temp',
    'normalize_percentile',
    'normalize',
    'z_score',
    'normal_clip',
    'normalize_percentile_z_score',
    'min_max',
    'normalize_constant',
    'save_configs',
    'binary_normal',
    '_raise',
    'GetPSF',
    'add_delimiter',
    'get_3d_stack',

]

def _raise(e):
    raise e


def GetPSF(psf_path,d1=0,d2=61,padding_size=None,img_size=None):
    def PSF2list(psf,CAindex,padding_size):
        psf_h,psf_w,Nx,Ny,depth=psf.shape
        mmid =psf_h // 2
        Nnum=Nx
        psf_list=[]
        for d in range(depth):
            si = CAindex[d, 0]
            ei = CAindex[d, 1]+1
            for i in range(Nnum):
                for j in range(Nnum):
                    psf_padding=np.zeros([padding_size,padding_size],np.float32)
                    sub_psf = np.squeeze(psf[si:ei,si:ei, i, j, d])
                    psf_shift = np.fft.ifftshift(sub_psf)
                    mmid = psf_shift.shape[0] // 2
                    psf_padding[0:mmid,0:mmid]=psf_shift[0:mmid,0:mmid]
                    psf_padding[0:mmid,-mmid:]=psf_shift[0:mmid,-mmid:]
                    psf_padding[-mmid:,0:mmid]=psf_shift[-mmid:,0:mmid]
                    psf_padding[-mmid:,-mmid:]=psf_shift[-mmid:,-mmid:]
                    psf_list.append(psf_padding)
        return psf_list


    data_dict = mat73.loadmat(psf_path)
    LFpsf = data_dict['H']
    CAindex = np.array(data_dict['CAindex'] - 1, np.int16)
    psf_ReArrange = PSF2list(psf=LFpsf, CAindex=CAindex, padding_size=padding_size)
    psf_ReArrange = np.stack(psf_ReArrange,axis=0)
    return  np.float32(psf_ReArrange)


def save_configs(save_folder,cg):
    configs = {key: value for key, value in cg.__dict__.items() if not (key.startswith('__') or key.startswith('_'))}
    np.save(os.path.join(save_folder, 'training_configs'),configs)

def get_2d_lf(filename, path, normalize_fn, **kwargs):
    def _LFP2ViewMap(img, angRes):
        img = np.squeeze(img)
        h, w = img.shape
        base_h = h // angRes
        base_w = w // angRes
        VP_ = np.zeros(img.shape, np.float32)
        for v in range(angRes):
            for u in range(angRes):
                VP_[v * base_h:(v + 1) * base_h, u * base_w:(u + 1) * base_w] = img[v::angRes, u::angRes]
        return VP_

    def _ViewMap2LFP(img, angRes):
        img = np.squeeze(img)
        h, w = img.shape
        base_h = h // angRes
        base_w = w // angRes
        LFP_ = np.zeros(img.shape, np.float32)
        for v in range(angRes):
            for u in range(angRes):
                LFP_[v::angRes, u::angRes] = img[v * base_h:(v + 1) * base_h, u * base_w:(u + 1) * base_w]
        return LFP_

    def _identity(img, angRes):
        return img

    # image = imageio.imread(path + filename).astype(np.uint16)
    image = imageio.imread(os.path.join(path,filename))
    if 'read_type' in kwargs:
        read_type = kwargs['read_type']
    else:
        read_type = None

    if read_type is not None:
        assert 'ViewMap' in read_type or 'LFP' in read_type, 'wrong img type'
        if '1' in read_type:
            trans_func = _identity if 'LFP' in read_type else _ViewMap2LFP
        elif '2' in read_type:
            trans_func = _identity if 'ViewMap' in read_type else _LFP2ViewMap
        else:
            raise Exception('wrong img type')
        image = trans_func(image, angRes=kwargs['angRes'])

    image = image[:, :, np.newaxis] if image.ndim == 2 else image


    if normalize_fn is not None:
        return normalize_fn(image)
    else:
        return image


def get_img3d_fn(filename, path, normalize_fn):
    """
    Parames:
        mode - Depth : read 3-D image in format [depth=slices, height, width, channels=1]
               Channels : [height, width, channels=slices]
    """
    image = imageio.volread(path + filename)  # [depth, height, width]
    # image = image[..., np.newaxis] # [depth, height, width, channels]
    if normalize_fn is not None:
        return normalize_fn(image)
    else:
        return image

def get_3d_stack(filename,path,normalize_fn):

    img = io.imread(os.path.join(path, filename))

    if (img.dtype != np.float32):
        img = img.astype(np.float32, casting='unsafe')
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    if normalize_fn is not None:
        return normalize_fn(img)
    else:
        return img


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
    height, width, depth = image.shape
    image_re = np.zeros([depth, height, width])
    for d in range(depth):
        image_re[d, :, :] = image[:, :, d]
    return image_re


def get_and_rearrange3d(filename, path, normalize_fn):
    image = get_img3d_fn(filename, path, normalize_fn=normalize_fn)
    return rearrange3d_fn(image)


def get_img2d_fn(filename, path, normalize_fn, **kwargs):
    image = imageio.imread(os.path.join(path , filename)).astype(np.uint16)
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    # print(image.shape)
    return normalize_fn(image, **kwargs)


def get_lf_extra(filename, path, n_num, normalize_fn, padding=False, **kwargs):
    image = get_img2d_fn(filename, path, normalize_fn, **kwargs)
    extra = lf_extract_fn(image, n_num=n_num, padding=padding)
    return extra


def normalize(x):
    max_ = np.max(x) * 1.1
    # max_ = 255.
    # max_ = np.max(x)
    x = x / (max_ / 2.)
    x = x - 1
    return x


def normalize_constant(im):
    assert im.dtype in [np.uint8, np.uint16]
    x = im.astype(np.float)
    max_ = 255. if im.dtype == np.uint8 else 65536.
    # x = x / (max_ / 2.) - 1.
    x = x / (max_)
    return x


def z_score(x):
    std_ = np.std(x)
    mean_ = np.mean(x)
    return (x - mean_) / std_


def min_max(x, eps=1e-7):
    max_ = np.max(x)
    min_ = np.min(x)
    return (x - min_) / (max_ - min_ + eps)


def normalize_percentile(im, low=0, high=100, clip=False, is_random=False,**kwargs):
    if is_random:
        _p_low = np.random.uniform(0.1, 0.5)
        p_low = np.percentile(im, _p_low)

        _p_high = np.random.uniform(99.5, 99.9)
        p_high = np.percentile(im, _p_high)
    else:
        p_low = np.percentile(im, low)
        p_high = np.percentile(im, high)
    eps = 1e-7
    x = ((im - p_low) / (p_high - p_low + eps)).astype(np.float32)
    if clip:
        # x[x>1.0]=1.0
        x[x < .0] = .0
    if 'gamma' in kwargs:
        gamma = kwargs['gamma']
        x=np.power(x, gamma)
    # return x
    return x.astype(np.float32)



def normalize_percentile_z_score(im, low=0.2, high=99.8):
    p_low = np.percentile(im, low)
    p_high = np.percentile(im, high)
    eps = 1e-7
    x = np.clip(im, p_low, p_high)
    mean_ = np.mean(x)
    std = np.std(x)
    return (x - mean_) / std


def binary_normal(x):
    # max_ = np.max(x)
    max_ = 255.
    # max_ = np.max(x)
    x = x / max_

    return x


def resize_fn(x, size):
    '''
    Param:
        -size: [height, width]
    '''
    x = np.array(pilimg.fromarray(x).resize(size=(size[1], size[0]), resample=pilimg.BICUBIC))

    return x


def lf_extract_fn(lf2d, n_num=11, mode='toChannel', padding=False):
    """
    Extract different views from a single LF projection

    Params:
        -lf2d - 2-D light field projection
        -mode - 'toDepth' -- extract views to depth dimension (output format [depth=multi-slices, h, w, c=1])
                'toChannel' -- extract views to channel dimension (output format [h, w, c=multi-slices])
        -padding -   True : keep extracted views the same size as lf2d by padding zeros between valid pixels
                     False : shrink size of extracted views to (lf2d.shape / Nnum);
    Returns:
        ndarray [height, width, channels=n_num^2] if mode is 'toChannel'
                or [depth=n_num^2, height, width, channels=1] if mode is 'toDepth'
    """
    n = n_num
    h, w, c = lf2d.shape
    if padding:
        if mode == 'toDepth':
            lf_extra = np.zeros([n * n, h, w, c])  # [depth, h, w, c]

            d = 0
            for i in range(n):
                for j in range(n):
                    lf_extra[d, i: h: n, j: w: n, :] = lf2d[i: h: n, j: w: n, :]
                    d += 1
        elif mode == 'toChannel':
            lf2d = np.squeeze(lf2d)
            lf_extra = np.zeros([h, w, n * n])

            d = 0
            for i in range(n):
                for j in range(n):
                    lf_extra[i: h: n, j: w: n, d] = lf2d[i: h: n, j: w: n]
                    d += 1
        else:
            raise Exception('unknown mode : %s' % mode)
    else:
        new_h = int(np.ceil(h / n))
        new_w = int(np.ceil(w / n))

        if mode == 'toChannel':

            lf2d = np.squeeze(lf2d)
            lf_extra = np.zeros([new_h, new_w, n * n])

            d = 0
            for i in range(n):
                for j in range(n):
                    lf_extra[:, :, d] = lf2d[i: h: n, j: w: n]
                    d += 1

        elif mode == 'toDepth':
            lf_extra = np.zeros([n * n, new_h, new_w, c])  # [depth, h, w, c]

            d = 0
            for i in range(n):
                for j in range(n):
                    lf_extra[d, :, :, :] = lf2d[i: h: n, j: w: n, :]
                    d += 1
        else:
            raise Exception('unknown mode : %s' % mode)

    return lf_extra


def do_nothing(x):
    return x


def normal_clip(x, low=1, high=100):
    min_ = np.percentile(x, low)
    max_ = np.max(x) if high == 100 else np.percentile(x, high)
    x = np.clip(x, min_, max_)

    # return binary_normal(x)
    return min_max(x)


def _write3d(x, path, bitdepth=8, clip=True):
    """
    x : [depth, height, width, channels=1]
    """
    assert (bitdepth in [8, 16, 32])
    max_ = 1.2 * np.max(x)
    if clip:
        x = np.clip(x, 0, max_)
    if bitdepth == 32:
        x = x.astype(np.float32)

    else:
        # x = _clip(x, 0.2)
        min_ = np.min(x)
        x = (x - min_) / (max_ - min_)
        # x[:,:16,:16,:],x[:,-16:,-16:,:]=0,0
        # x[:,-16:,:16,:],x[:,:16,-16:,:]=0,0

        if bitdepth == 8:
            x = x * 255
            x = x.astype(np.uint8)
        else:
            x = x * 65535
            x = x.astype(np.uint16)

    imageio.volwrite(path, x[..., 0])


def write3d(x, path, bitdepth=32):
    """
    x : [batch, depth, height, width, channels] or [batch, height, width, channels>3]
    """

    # print(x.shape)
    dims = len(x.shape)

    if dims == 4:
        batch, height, width, n_channels = x.shape
        x_re = np.zeros([batch, n_channels, height, width, 1])
        for d in range(n_channels):
            slice = x[:, :, :, d]
            x_re[:, d, :, :, :] = slice[:, :, :, np.newaxis]

    elif dims == 5:
        x_re = x
    else:
        raise Exception('unsupported dims : %s' % str(x.shape))

    batch = x_re.shape[0]
    if batch == 1:
        _write3d(x_re[0], path, bitdepth)
    else:
        fragments = path.split('.')
        new_path = ''
        for i in range(len(fragments) - 1):
            new_path = new_path + fragments[i]
        for index, image in enumerate(x_re):
            # print(image.shape)
            _write3d(image, new_path + '_' + str(index) + '.' + fragments[-1], bitdepth)

# def write3d_temp(x, path, bitdepth=8):
#     x = np.clip(x, a_min=0, a_max=1)
#     x = np.squeeze((x - np.aminx)) / ((np.amax(x) - np.amin(x))) * 255
#     save_tiff_imagej_compatible(path, x.astype(np.uint8), axes='YXZ')


def add_delimiter(input_data, real_input):
    with open(input_data, 'r', encoding="utf-8") as fr:
        last_cursor= len(fr.readlines()) - 1
    with open(input_data, 'r', encoding="utf-8") as fr:
        with open(real_input, 'w', encoding="utf-8") as fw:
            for idx,line in enumerate(fr):
                if line == "}\n" and idx != last_cursor:
                    fw.writelines(line.strip("\n") + ',' + "\n")
                else:
                    fw.writelines(line)
    with open(real_input, 'r+', encoding="utf-8") as fs:
        content = fs.read()
        fs.seek(0)
        fs.write("[")
        fs.write(content)
        fs.seek(0, 2)
        fs.write("]")
