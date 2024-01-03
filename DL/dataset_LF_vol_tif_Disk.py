import tensorlayer as tl
import numpy as np
import os

from utils import *
import PIL.Image as pilImg

pre_shuffle=False

class Dataset_disk:
    def __init__(self, base_path, data_str, n_slices, n_num, lf2d_base_size, normalize_mode='max',
                 shuffle_for_epoch=True, multi_scale=False, **kwargs):
        '''
        Params:
            n_slices      : int, depth of the 3d target images (the reconstructions)
            n_num         : int, Nnum of light filed imaging
            lf2d_base_size: 2-element list, [height, width], equals to (lf2d_size / n_num)
            normalize_mode: str, normalization mode of dataset in ['max', 'percentile']
            shuffle       : boolean, whether to shuffle the training dataset
            multi_scale   : boolean, whether to generate multi-scale HRs
        '''
        self.base_path = base_path
        self.data_str = data_str

        self.lf2d_base_size = lf2d_base_size
        self.n_slices = n_slices
        self.n_num = n_num
        self.multi_scale = multi_scale
        self.shuffle_all_data = False
        self.save_hdf5 = True
        if normalize_mode == 'normalize_percentile_z_score':
            self.normalize_fn = normalize_percentile_z_score
        elif normalize_mode == 'percentile':
            self.normalize_fn = normalize_percentile
        elif normalize_mode == None:
            self.normalize_fn = None
        elif normalize_mode == 'constant':
            self.normalize_fn = normalize_constant
        else:
            self.normalize_fn = normalize

        self.shuffle_for_epoch = shuffle_for_epoch
        self.sample_ratio = 1.0
        self.train_lf2d_path=os.path.join(self.base_path,self.data_str[1])
        self.train_hr3d_path=os.path.join(self.base_path, self.data_str[0])
        # define mask path
        self.update_parameters(allow_new=True, **kwargs)


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
        def _shuffle_in_unison(arr1, arr2):
            """shuffle elements in arr1 and arr2 in unison along the leading dimension
            Params:
                -arr1, arr2: np.ndarray
                    must be in the same size in the leading dimension
            """
            assert (len(arr1) == len(arr2))
            new_idx = np.random.permutation(len(arr1))
            return arr1[new_idx], arr2[new_idx]

        def _load_imgs(path, fn, regx='.*.tif', printable=False, type_name=None, **kwargs ):
            img_list = sorted(tl.files.load_file_list(path=path, regx=regx, printable=printable))
            list_len = int(len(img_list) * self.sample_ratio)

            return img_list[0:list_len]

        ###loading

        print('sample ratio: %0.2f' % self.sample_ratio)
        training_hr3d_list = _load_imgs(self.train_hr3d_path,
                                fn=get_3d_stack,
                                normalize_fn=self.normalize_fn, type_name='HR')
        training_lf2d_list= _load_imgs(self.train_lf2d_path,
                                          fn=get_2d_lf, n_num=self.n_num,normalize_fn=self.normalize_fn, type_name='LFP')

        self.mask_num = 0

        ##check
        print('-----------ToDisk-----------')
        print('sample ratio: %0.2f' % self.sample_ratio)

        if (len(training_hr3d_list) == 0) or (len(training_lf2d_list) == 0):
            raise Exception("none of the images have been loaded, please check the file directory in config")
        assert len(training_hr3d_list) == len(training_lf2d_list)

        ## assign
        # [self.training_hr3d, self.training_lf2d] = _shuffle_in_unison(training_hr3d, training_lf2d) if shuffle else [training_hr3d, training_lf2d]
        [self.training_hr3d_list, self.training_lf2d_list] = training_hr3d_list, training_lf2d_list

        self.training_pair_num = len(self.training_hr3d_list)



    def prepare(self, batch_size, n_epochs):
        '''
        this function must be called after the Dataset instance is created
        '''

        self._load_dataset()

        self.test_img_num = int(self.training_pair_num * 0.1)


        self.batch_size = batch_size
        self.n_epochs = n_epochs

        self.cursor = self.test_img_num
        self.epoch = 0
        print('\nHR dataset : %d\nLF dataset: %d\nHR_mask dataset: %d\n' % (
        len(self.training_hr3d_list), len(self.training_lf2d_list), self.mask_num))

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
        lf_img=[]
        target3d=[]
        for f_idx in self.data_shuffle_matrix[0,0:n]:
            lf_img.append(get_2d_lf(filename=self.training_lf2d_list[f_idx],path=self.train_lf2d_path,n_num=self.n_num,normalize_fn=self.normalize_fn))
            target3d.append(get_3d_stack(filename=self.training_hr3d_list[f_idx],path=self.train_hr3d_path,normalize_fn=self.normalize_fn))
        return np.asarray(target3d), \
               np.asarray(lf_img), \
               # self.training_hr3d_list[0:n],\
               # self.training_lf2d_list[0:n]


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
            syn_view = []
            target3d = []
            for f_idx in shuffle_idx:
                lf_img.append(
                    get_2d_lf(filename=self.training_lf2d_list[f_idx], path=self.train_lf2d_path, n_num=self.n_num,normalize_fn=self.normalize_fn)
                              )
                target3d.append(
                    get_3d_stack(filename=self.training_hr3d_list[f_idx], path=self.train_hr3d_path,normalize_fn=self.normalize_fn)
                                )
            return np.asarray(target3d), np.asarray(lf_img), idx - nt, self.epoch
        raise Exception('epoch index out of bounds:%d/%d' % (self.epoch, self.n_epochs))