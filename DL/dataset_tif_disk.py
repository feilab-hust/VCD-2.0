import tensorlayer as tl
import numpy as np
import os

from utils import *
import PIL.Image as pilImg
from config import config


class Dataset_tif_disk:
    def __init__(self,
                 Target3D_path,
                 Scan_view_path,
                 Synth_view_path,
                 LFP_path,
                 n_num, lf2d_base_size, normalize_mode='max', shuffle_for_epoch=True, multi_scale=False, **kwargs):
        '''
        Params:
            n_slices      : int, depth of the 3d target images (the reconstructions)
            n_num         : int, Nnum of light filed imaging
            lf2d_base_size: 2-element list, [height, width], equals to (lf2d_size / n_num)
            normalize_mode: str, normalization mode of dataset in ['max', 'percentile']
            shuffle       : boolean, whether to shuffle the training dataset
            multi_scale   : boolean, whether to generate multi-scale HRs
        '''

        self.Target3D_path = Target3D_path
        self.Scan_iew_path = Scan_view_path
        self.Synth_view_path = Synth_view_path
        self.LFP_path = LFP_path
        self.n_num = n_num
        self.shuffle_all_data = False
        self.shuffle_for_epoch = shuffle_for_epoch
        self.sample_ratio = 1.0

        if normalize_mode == 'normalize_percentile_z_score':
            self.normalize_fn = normalize_percentile_z_score
        elif normalize_mode == 'percentile':
            self.normalize_fn = normalize_percentile
        elif normalize_mode == 'constant':
            self.normalize_fn = normalize_constant
        else:
            self.normalize_fn = normalize

        self.update_parameters(allow_new=True, **kwargs)

        # #define mask path
        # from config_trans import mask_path
        # self.using_binary_loss=False

        # if self.using_binary_loss:
        #     self.hr_mask_path=mask_path
        #     self.binary_norm=binary_normal

    def update_parameters(self, allow_new=False, **kwargs):
        if not allow_new:
            attr_new = []
            for k in kwargs:
                try:
                    getattr(self, k)
                except AttributeError:
                    attr_new.append(k)
            if len(attr_new) > 0:
                raise AttributeError("Not allowed to add new parameters (%s)" % ', '.join(attr_new))
        for k in kwargs:
            setattr(self, k, kwargs[k])

    def _load_dataset(self, shuffle=True):


        def _load_imgs(path, fn, regx='.*.tif', printable=False, type_name=None, **kwargs, ):
            img_list = sorted(tl.files.load_file_list(path=path, regx=regx, printable=printable))
            imgs = []

            list_len = int(len(img_list) * self.sample_ratio)
            # for img_file in img_list[0:list_len]:
            #     img = fn(img_file, path, **kwargs)
            #     if (img.dtype != np.float32):
            #         img = img.astype(np.float32, casting='unsafe')
            #     print('\r%s training data loading: %s -- %s  ---min: %f max:%f' % (
            #     type_name, img_file, str(img.shape), np.min(img), np.max(img)), end='')
            #     imgs.append(img)
            return imgs, img_list[0:list_len]

        def _load_imgs_1(path, fn, regx='.*.tif', printable=False, type_name=None, **kwargs, ):
            img_list = sorted(tl.files.load_file_list(path=path, regx=regx, printable=printable))
            imgs = []
            # from skimage import io
            list_len = int(len(img_list) * self.sample_ratio)
            # for img_file in img_list[0:list_len]:
            #     img = io.imread(os.path.join(path, img_file))
            #     # img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
            #     if (img.dtype != np.float32):
            #         img = img.astype(np.float32, casting='unsafe')
            #     if img.ndim == 3:
            #         img = np.transpose(img, (1, 2, 0))
            #     if img.ndim == 2:
            #         img = np.expand_dims(img, axis=-1)
            #     print('\r%s training data loading: %s -- %s  ---min: %f max:%f' % (
            #     type_name, img_file, str(img.shape), np.min(img), np.max(img)), end='')
            #     imgs.append(img)
            return imgs, img_list[0:list_len]

        ###loading
        print('sample ratio: %0.2f' % self.sample_ratio)
        # training_Target3D,training_Target3D_list=_load_imgs(self.Target3D_path, fn=get_and_rearrange3d,
        #                                                     normalize_fn=self.normalize_fn,type_name='Target3D')
        _, self.training_Target3D_list = _load_imgs_1(self.Target3D_path, fn=get_and_rearrange3d,
                                                                           normalize_fn=self.normalize_fn,
                                                                           type_name='Target3D')
        _, self.training_ScanView_list = _load_imgs(self.Scan_iew_path, fn=get_2d_lf,
                                                                         normalize_fn=self.normalize_fn,
                                                                         type_name='ScanVIew',
                                                                         read_type=config.preprocess.SynView_type,
                                                                         angRes=self.n_num)
        _, self.training_SynView_list = _load_imgs(self.Synth_view_path, fn=get_2d_lf,
                                                                       normalize_fn=self.normalize_fn,
                                                                       type_name='SynView',
                                                                       read_type=config.preprocess.SynView_type,
                                                                       angRes=self.n_num)
        _, self.training_lf2d_list = _load_imgs(self.LFP_path, fn=get_2d_lf, n_num=self.n_num,
                                                                 normalize_fn=self.normalize_fn, type_name='LFP',
                                                                 read_type=config.preprocess.LFP_type,
                                                                 angRes=self.n_num)
        # if self.shuffle_all_data:
        #     [training_Target3D,training_Target3D_list,training_SynView,training_SynView_list,training_lf2d,training_lf2d_list]=_shuffle_in_unison(
        #                                                                                                 [training_Target3D,training_Target3D_list,
        #                                                                                                  training_SynView,training_SynView_list,
        #                                                                                                   training_lf2d,training_lf2d_list])
        # print('\n[!]save dataset order')
        # assert training_SynView_list==training_lf2d_list,print('the filename of LF and WF must be  same ')

        ##check
        if (len(self.training_Target3D_list) == 0) or (len(self.training_lf2d_list) == 0):
            raise Exception("none of the images have been loaded, please check the file directory in config")
        assert len(self.training_Target3D_list) == len(self.training_lf2d_list)

        self.training_pair_num = len(self.training_SynView_list)

    def prepare(self, batch_size, n_epochs,skip_loading=False):
        '''
        this function must be called after the Dataset instance is created
        '''
        if not skip_loading:
            if os.path.exists(self.LFP_path) and os.path.exists(self.Synth_view_path):
                self._load_dataset()
            else:
                raise Exception('image data path doesn\'t exist')

        self.test_img_num = int(self.training_pair_num * 0.03)

        self.batch_size = batch_size
        self.n_epochs = n_epochs

        self.cursor = self.test_img_num
        self.epoch = 0
        print('\nTarget3D dataset : %d\nSynView dataset : %d\nLF dataset: %d\n' % (
        len(self.training_Target3D_list), len(self.training_ScanView_list), len(self.training_SynView_list)))

        data_shuffle_matrix = []

        for idx in range(self.n_epochs + 1):
            temp = np.arange(0, self.test_img_num, dtype=np.int32)
            temp = np.append(temp,
                             np.random.permutation(self.training_pair_num - self.test_img_num) + self.test_img_num)

            if self.shuffle_for_epoch == True:
                data_shuffle_matrix.append(temp)
            else:
                temp.sort()
                data_shuffle_matrix.append(temp)

        self.data_shuffle_matrix = np.stack(data_shuffle_matrix, axis=0)
        return self.training_pair_num - self.test_img_num

    def for_test(self):
        n = self.test_img_num

        n = self.test_img_num
        lf_img=[]
        clf_img=[]
        slf_img=[]
        target3d=[]
        for f_idx in self.data_shuffle_matrix[0,0:n]:
            lf_img.append(get_2d_lf(filename=self.training_lf2d_list[f_idx],path=self.LFP_path,n_num=self.n_num,normalize_fn=self.normalize_fn))
            clf_img.append(
                get_2d_lf(filename=self.training_SynView_list[f_idx], path=self.Synth_view_path, n_num=self.n_num,
                          normalize_fn=self.normalize_fn))
            slf_img.append(
                get_2d_lf(filename=self.training_ScanView_list[f_idx], path=self.Scan_iew_path, n_num=self.n_num,
                          normalize_fn=self.normalize_fn))
            target3d.append(get_3d_stack(filename=self.training_Target3D_list[f_idx],path=self.Target3D_path))
        return np.asarray(target3d), \
               np.asarray(slf_img), \
               np.asarray(clf_img), \
               np.asarray(lf_img), \

    def hasNext(self):
        return True if self.epoch < self.n_epochs else False

    def iter(self):
        '''
        return the next batch of the training data
        '''
        nt = self.test_img_num
        if self.epoch < self.n_epochs:
            if self.cursor + self.batch_size > self.training_pair_num:
                self.epoch += 1
                self.cursor = nt

            idx = self.cursor
            end = idx + self.batch_size
            self.cursor += self.batch_size
            shuffle_idx = self.data_shuffle_matrix[self.epoch][idx:end]

            lf_img = []
            clf_img = []
            slf_img = []
            target3d = []
            for f_idx in shuffle_idx:
                lf_img.append(get_2d_lf(filename=self.training_lf2d_list[f_idx], path=self.LFP_path, n_num=self.n_num,
                                        normalize_fn=self.normalize_fn))
                clf_img.append(
                    get_2d_lf(filename=self.training_SynView_list[f_idx], path=self.Synth_view_path, n_num=self.n_num,
                              normalize_fn=self.normalize_fn))
                slf_img.append(
                    get_2d_lf(filename=self.training_ScanView_list[f_idx], path=self.Scan_iew_path, n_num=self.n_num,
                              normalize_fn=self.normalize_fn))
                target3d.append(get_3d_stack(filename=self.training_Target3D_list[f_idx], path=self.Target3D_path))

            return np.asarray(target3d), \
                   np.asarray(slf_img), \
                   np.asarray(clf_img), \
                   np.asarray(lf_img), \
                   idx - nt, \
                   self.epoch, \
                # [self.training_Target3D[i] for i in shuffle_idx],\
            # [self.training_SynView_list[i] for i in shuffle_idx]

        raise Exception('epoch index out of bounds:%d/%d' % (self.epoch, self.n_epochs))