def is_number(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


class Trainer:
    def __init__(self, dataset):
        self.dataset = dataset
        self.losses = {}
        self.fetches={}
        self.pre_losses={}
        self.pre_fetches = {}
        self.losses2 = {}

    def build_graph(self):
        ###========================== DEFINE MODEL ============================###
        with tf.variable_scope('learning_rate'):
            self.learning_rate = tf.Variable(lr_init, trainable=False)
            self.pre_learning_rate= tf.Variable(local_pre_SRVCD_dict['lr_init'], trainable=False,name='preTrain_learningrate')
        # recon_net
        denoise_tag = local_configs.net_setting.denoise_model
        ngf1 = local_configs.net_setting.ngf[0]
        SR_tag = local_configs.net_setting.SR_model
        ngf2 = local_configs.net_setting.ngf[1]
        Recon_tag = local_configs.net_setting.Recon_model
        ngf3 = local_configs.net_setting.ngf[2]
        print('[!] Denoise:%s --- ngf:%d \n[!] SR:%s --- ngf:%d\n[!] Recon:%s --- ngf:%d'%(denoise_tag,ngf1,SR_tag,ngf2,Recon_tag,ngf3))
        denoise_model = eval(denoise_tag)
        sr_model = eval(SR_tag)
        recon_model = eval(Recon_tag)

        # net_tag = local_configs.net_tag
        # input_size = np.array([img_size, img_size])* n_num if 'SA' in SR_tag else np.array([img_size, img_size])
        input_size = np.array([img_size, img_size])
        SR_size = input_size * sr_factor
        Recon_size = np.multiply(SR_size, ReScale_factor)

        self.plchdr_lf = tf.placeholder('float32', [batch_size, *input_size, 1], name='t_LFP')
        self.plchdr_SynView = tf.placeholder('float32', [batch_size, *input_size, 1], name='t_SynView')
        self.plchdr_Scan_View = tf.placeholder('float32', [batch_size, *SR_size, 1], name='t_Scan_View')
        self.plchdr_Target3D = tf.placeholder('float32', [batch_size, *Recon_size, n_slices], name='t_Target3D')

        with tf.device('/gpu:{}'.format(local_configs.TRAIN.device)):
            self.denoise_net = denoise_model (LFP=self.plchdr_lf, output_size=input_size, sr_factor=1, angRes=n_num,reuse=False,channels_interp=ngf1,name=denoise_tag)

            self.SR_net = sr_model(LFP=self.denoise_net.outputs, output_size=SR_size, sr_factor=sr_factor,
                                   angRes=n_num, reuse=False, name=SR_tag, channels_interp=ngf2,
                                   normalize_mode=normalize_mode, transform_layer='SAI2Macron')
            self.pre_SR_net = sr_model(LFP=self.plchdr_SynView, output_size=SR_size, sr_factor=sr_factor,
                                   angRes=n_num, reuse=True, name=SR_tag, channels_interp=ngf2,
                                   normalize_mode=normalize_mode, transform_layer='SAI2Macron')

            self.Recon_net = recon_model(lf_extra=self.SR_net.outputs, n_slices=n_slices, output_size=Recon_size,
                                         is_train=True, reuse=False, name=Recon_tag, channels_interp=ngf3,
                                         normalize_mode=normalize_mode,transform='SAI2ViewStack',pyrimid_list=local_configs.net_setting.Unetpyrimid_list)
            self.pre_Recon_net = recon_model(lf_extra=self.pre_SR_net.outputs, n_slices=n_slices, output_size=Recon_size,
                                         is_train=True, reuse=True, name=Recon_tag, channels_interp=ngf3,
                                         normalize_mode=normalize_mode,transform='SAI2ViewStack',pyrimid_list=local_configs.net_setting.Unetpyrimid_list)
        self.denoise_net.print_params(False)
        self.SR_net.print_params(False)
        self.Recon_net.print_params(False)

        denoise_vars = tl.layers.get_variables_with_name(denoise_tag, train_only=True, printable=False)
        SR_vars = tl.layers.get_variables_with_name(SR_tag, train_only=True, printable=False)
        Recon_vars = tl.layers.get_variables_with_name(Recon_tag, train_only=True, printable=False)
        # ====================
        # loss function
        # =====================
        self.loss = 0  # initial
        # self._get_losses()    # get losses
        self.denoise_loss = 0
        self.SR_loss = 0
        self.Recon_loss = 0

        # define SR loss

        for key in denoise_loss:
            temp_func = eval(key)
            temp_loss = temp_func(image=self.denoise_net.outputs, reference=self.plchdr_SynView)
            self.denoise_loss = self.denoise_loss + denoise_loss[key] * temp_loss
            self.losses.update({'Denoise_' + key: denoise_loss[key] * temp_loss})
            tf.summary.scalar(key, temp_loss)

        for key in SR_loss:
            temp_func = eval(key)
            temp_loss = temp_func(image=self.SR_net.outputs, reference=self.plchdr_Scan_View)
            self.SR_loss = self.SR_loss + SR_loss[key] * temp_loss
            self.losses.update({'SR_' + key: SR_loss[key] * temp_loss})
            # self.losses1.update({'SR_' + key: SR_loss[key] * temp_loss})
            tf.summary.scalar(key, temp_loss)

        for key in Recon_loss:
            temp_func = eval(key)
            temp_loss = temp_func(image=self.Recon_net.outputs, reference=self.plchdr_Target3D)
            self.Recon_loss = self.Recon_loss + Recon_loss[key] * temp_loss
            self.losses.update({'Recon_' + key: Recon_loss[key] * temp_loss})
            # self.losses1.update({'Recon_' + key: Recon_loss[key] * temp_loss})
            # self.losses2.update({'Recon_' + key: Recon_loss[key] * temp_loss})
            tf.summary.scalar(key, temp_loss)


        #self.loss_stage1 = loss_ratio[0] * self.denoise_loss + loss_ratio[1] * self.SR_loss
        #self.loss_stage2 = loss_ratio[0] * self.SR_loss + loss_ratio[1] * self.Recon_loss
        self.loss_stage1 = self.denoise_loss
        self.loss_stage2 = loss_ratio[0] * self.denoise_loss + loss_ratio[1] * self.SR_loss
        self.loss_stage3 = loss_ratio[0] * self.denoise_loss + loss_ratio[1] * self.SR_loss + loss_ratio[2] * self.Recon_loss
        #self.loss = 0.1*self.denoise_loss + 0.3*self.SR_loss + 0.6*self.Recon_loss
        tf.summary.scalar('learning_rate', self.learning_rate)
        # define test_loss when test
        self.loss_test = loss_ratio[0] * self.denoise_loss + loss_ratio[1] * self.SR_loss + loss_ratio[2] * self.Recon_loss
        # ----------------create sess-------------
        configProto = tf.ConfigProto(allow_soft_placement=False, log_device_placement=False)
        configProto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=configProto)

        # self.pre_train_opt =
        # self.den_opt = tf.train.AdamOptimizer(self.learning_rate, beta1=beta1).minimize(self.denoise_loss,
        #                                                                                 var_list=denoise_vars)
        # self.SR_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=beta1).minimize(self.SR_loss, var_list=SR_vars)
        #
        # self.vcd_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=beta1).minimize(self.Recon_loss,
        #                                                                                   var_list=Recon_vars)

        self.fuse1_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=beta1).minimize(self.loss_stage1,
                                                                                            var_list=denoise_vars )
        self.fuse2_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=beta1).minimize(self.loss_stage2,
                                                                                            var_list=denoise_vars + SR_vars)
        self.fuse3_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=beta1).minimize(self.loss_stage3,
                                                                                            var_list=denoise_vars + SR_vars + Recon_vars)


        self.merge_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(log_dir, self.sess.graph)



        # update training fetches
        # for key in self.losses:
        #     self.fetches.update({key:self.losses[key]})
        self.fetches=self.losses
        self.fetches['opti_fuse_stage1'] = self.fuse1_optim
        self.fetches['opti_fuse_stage2'] = self.fuse2_optim
        self.fetches['opti_fuse_stage3'] = self.fuse3_optim
        self.fetches['batch_summary'] = self.merge_op

        # update pretrain fetches
        pre_sr_loss = mse_loss(image=self.pre_SR_net.outputs, reference=self.plchdr_Scan_View) + 0.1* EPI_mse_loss(image=self.pre_SR_net.outputs, reference=self.plchdr_Scan_View)
        pre_recon_loss = mse_loss(image=self.pre_Recon_net.outputs, reference=self.plchdr_Target3D) + 0.1 * edge_loss(image=self.pre_Recon_net.outputs, reference=self.plchdr_Target3D)
        self.pre_loss= pre_sr_loss*1+ pre_recon_loss*10
        pre_SR_optim = tf.train.AdamOptimizer(self.pre_learning_rate, beta1=beta1).minimize(pre_sr_loss, var_list=SR_vars)
        pre_Recon_optim = tf.train.AdamOptimizer(self.pre_learning_rate, beta1=beta1).minimize(self.pre_loss, var_list=SR_vars+Recon_vars)


        self.pre_losses.update(
            {'pre_sr_loss':pre_sr_loss,
             'pre_recon_loss': pre_recon_loss,
             }
        )
        self.pre_fetches=self.pre_losses
        # for key in self.pre_losses:
        #     self.pre_fetches.update({key:self.pre_losses[key]})

        self.pre_fetches['pre_SR_optim'] = pre_SR_optim
        self.pre_fetches['pre_Recon_optim'] = pre_Recon_optim

    def _train(self, begin_epoch):
        """Train the VCD-Net
        Params
            -begin_epoch: int, if not 0, a checkpoint file will be loaded and the training will continue from there
        """
        ## create folders to save result images and trained model

        tl.files.exists_or_mkdir(test_saving_dir)
        tl.files.exists_or_mkdir(checkpoint_dir)
        save_configs(save_folder=checkpoint_dir,cg=local_configs)

        tl.files.exists_or_mkdir(log_dir)
        tl.files.exists_or_mkdir(plot_test_loss_dir)
        tl.files.exists_or_mkdir(test_lf_dir)
        tl.files.exists_or_mkdir(test_hr_dir)
        tl.files.exists_or_mkdir(test_mr_dir)
        tl.files.exists_or_mkdir(test_stack_dir)
        tl.files.exists_or_mkdir(preSRVCD_ckpt_save)


        # initialize vars
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.assign(self.learning_rate, lr_init))

        # if loading_pretrain_model:

        denoise_pretrain_ckpt_dir = os.path.join(root_path, 'DL', local_configs.TRAIN.ckpt_dir, 'preDenoise')
        denoise_ckpt_file = [filename for filename in os.listdir(denoise_pretrain_ckpt_dir) if
                             ('.npz' in filename and 'best' in filename and 'denoise' in filename)]
        SR_pretrain_ckpt_dir=os.path.join(root_path,'DL',local_configs.TRAIN.ckpt_dir,'preViewSR')
        SR_ckpt_file = [filename for filename in os.listdir(SR_pretrain_ckpt_dir) if
                        ('.npz' in filename and 'best' in filename and 'SR' in filename)]
        recon_pretrain_ckpt_dir = os.path.join(root_path, 'DL', local_configs.TRAIN.ckpt_dir, 'preVCD')
        recon_ckpt_file= [filename for filename in os.listdir(recon_pretrain_ckpt_dir) if
                          ('.npz' in filename and 'best' in filename and 'recon' in filename) or ('.npz' in filename and 'best' in filename and 'vcd' in filename) ]

        if len(denoise_ckpt_file)!=0:
            tl.files.load_and_assign_npz(sess=self.sess, name=os.path.join(denoise_pretrain_ckpt_dir, denoise_ckpt_file[0]),
                                         network=self.denoise_net)
        if len(SR_ckpt_file) != 0:
            tl.files.load_and_assign_npz(sess=self.sess, name=os.path.join(SR_pretrain_ckpt_dir, SR_ckpt_file[0]),
                                         network=self.SR_net)
        if len(recon_ckpt_file) != 0:
            tl.files.load_and_assign_npz(sess=self.sess, name=os.path.join(recon_pretrain_ckpt_dir, recon_ckpt_file[0]),
                                         network=self.Recon_net)
        if (begin_epoch != 0):
            denoise_ckpt = self._traversal_through_ckpts(checkpoint_dir=checkpoint_dir, epoch=begin_epoch,
                                                         label='denoise')
            SR_ckpt = self._traversal_through_ckpts(checkpoint_dir=checkpoint_dir, epoch=begin_epoch, label='SR')
            Recon_ckpt = self._traversal_through_ckpts(checkpoint_dir=checkpoint_dir, epoch=begin_epoch, label='recon')
            assert SR_ckpt != None and Recon_ckpt != None, 'No ckpt has been found'
            tl.files.load_and_assign_npz(sess=self.sess, name=os.path.join(checkpoint_dir,denoise_ckpt), network=self.denoise_net)
            tl.files.load_and_assign_npz(sess=self.sess, name=os.path.join(checkpoint_dir,SR_ckpt), network=self.SR_net)
            tl.files.load_and_assign_npz(sess=self.sess, name=os.path.join(checkpoint_dir,Recon_ckpt), network=self.Recon_net)

        ###====================== LOAD DATA ===========================###
        ## Step 1 : pretrain SRVCD

        print(20*'-','Step1: pretrain SRVCD',20*'-')
        dataset_size = self.dataset.prepare(batch_size, local_pre_SRVCD_dict['n_epoch'])
        final_cursor = (dataset_size // batch_size - 1) * batch_size
        self._get_test_data()

        while self.dataset.hasNext():
            Stack_batch, Scan_batch, Syn_batch, LF_batch, cursor, epoch = self.dataset.iter()  # get data

            feed_pre_train = {
                self.plchdr_Target3D: Stack_batch,
                self.plchdr_Scan_View: Scan_batch,
                self.plchdr_SynView: Syn_batch,
            }

            epoch += begin_epoch
            step_time = time.time()

            # learning rate update
            if epoch != 0 and (epoch % local_pre_SRVCD_dict['decay_every'] == 0) and cursor == 0:
                new_lr_decay = local_pre_SRVCD_dict['lr_decay'] ** (epoch // local_pre_SRVCD_dict['decay_every'])
                self.sess.run(tf.assign(self.learning_rate, local_pre_SRVCD_dict['lr_init'] * new_lr_decay))
                print('\nlearning rate updated : %f\n' % (lr_init * new_lr_decay))

            # infer loss
            evaluated = self.sess.run(self.pre_fetches, feed_pre_train)

            # log
            loss_str = [name + ':' + str(value) for name, value in evaluated.items() if 'loss' in name]
            print("\rEpoch:[%d/%d] iter:[%d/%d] time: %4.3fs ---%s" % (
                epoch, local_pre_SRVCD_dict['n_epoch'], cursor, dataset_size, time.time() - step_time, loss_str), end='')

            if cursor == final_cursor:
                self._record_avg_test_pre_loss(epoch, self.sess)
                if epoch != 0 and (epoch % ckpt_saving_interval == 0):
                    self._save_intermediate_ckpt(epoch, self.sess)

        print('\n',20 * '-', 'Step2: Joint Training', 20 * '-')
        #
        # # update dataset
        dataset_size = self.dataset.prepare(batch_size, n_epoch, skip_loading=True)
        final_cursor = (dataset_size // batch_size - 1) * batch_size
        #
        # # update srNet and reconNet
        # sr_file_name = preSRVCD_ckpt_save + '/SR_net_{}.npz'.format('best')
        # recon_file_name = preSRVCD_ckpt_save + '/recon_net_{}.npz'.format('best')
        # tl.files.load_and_assign_npz(sess=self.sess, name=sr_file_name, network=self.SR_net)
        # tl.files.load_and_assign_npz(sess=self.sess, name=recon_file_name,network=self.Recon_net)

        while self.dataset.hasNext():

            Stack_batch, Scan_batch, Syn_batch, LF_batch, cursor, epoch = self.dataset.iter()  # get data

            feed_train = {
                self.plchdr_Target3D: Stack_batch,
                self.plchdr_Scan_View: Scan_batch,
                self.plchdr_SynView: Syn_batch,
                self.plchdr_lf: LF_batch,
            }


            epoch += begin_epoch
            step_time = time.time()

            # learning rate update
            if epoch != 0 and (epoch % decay_every == 0) and cursor == 0:
                new_lr_decay = lr_decay ** (epoch // decay_every)
                self.sess.run(tf.assign(self.learning_rate, lr_init * new_lr_decay))
                print('\nlearning rate updated : %f\n' % (lr_init * new_lr_decay))

            # infer loss
            evaluated = self.sess.run(self.fetches, feed_train)

            # log
            loss_str = [name + ':' + str(value) for name, value in evaluated.items() if 'loss' in name]
            print("\rEpoch:[%d/%d] iter:[%d/%d] time: %4.3fs ---%s" % (
                epoch, n_epoch + begin_epoch, cursor, dataset_size, time.time() - step_time, loss_str), end='')
            self.summary_writer.add_summary(evaluated['batch_summary'],
                                            epoch * (dataset_size // batch_size - 1) + cursor / batch_size)
            ##record and save checkpoints
            if cursor == final_cursor:
                self._record_avg_test_loss(epoch, self.sess)
                if epoch != 0 and (epoch % ckpt_saving_interval == 0):
                    self._save_intermediate_ckpt(epoch, self.sess)


    def _record_avg_test_pre_loss(self, epoch, sess):
        if 'Pre_min_test_loss' not in dir(self):
            self.Pre_min_test_loss = 1e10
            self.Pre_best_epoch = 0

        test_loss = 0
        test_data_num = len(self.test_LFP)
        print("")
        for idx in range(0, test_data_num, batch_size):
            if idx + batch_size <= test_data_num:
                test_synview_batch = self.test_Synview[idx: idx + batch_size]
                test_scanview_batch = self.test_Scan_View[idx: idx + batch_size]
                test_target3d_batch = self.test_target3d[idx: idx + batch_size]
                feed_test = {
                             self.plchdr_SynView: test_synview_batch,
                             self.plchdr_Scan_View: test_scanview_batch,
                             self.plchdr_Target3D: test_target3d_batch
                             }
                test_loss_batch, losses_batch = sess.run([self.pre_loss, self.pre_losses], feed_test)
                loss_str = [name + ':' + str(value) for name, value in losses_batch.items() if 'loss' in name]
                test_loss += test_loss_batch
                print('\rvalidation  [% 2d/% 2d] loss = %.6f --%s ' % (idx, test_data_num, test_loss_batch, loss_str),
                      end='')
        test_loss /= (len(self.test_LFP) // batch_size)
        print('\navg = %.6f best = %.6f (@epoch%d)' % (test_loss, self.Pre_min_test_loss, self.Pre_best_epoch))
        if (test_loss < self.Pre_min_test_loss):
            self.Pre_min_test_loss = test_loss
            self.Pre_best_epoch = epoch
            tag = 'best'
            sr_file_name = preSRVCD_ckpt_save + '/SR_net_{}.npz'.format(tag)
            recon_file_name = preSRVCD_ckpt_save + '/recon_net_{}.npz'.format(tag)

            tl.files.save_npz(self.pre_SR_net.all_params, name=sr_file_name, sess=sess)
            tl.files.save_npz(self.pre_Recon_net.all_params, name=recon_file_name, sess=sess)


    def _get_test_data(self):
        self.test_target3d, self.test_Scan_View, self.test_Synview, self.test_LFP = self.dataset.for_test()
        for i in range(test_num):
            write3d(self.test_target3d[i:i + 1], test_stack_dir + '/Target3d_%d.tif' % i)
            write3d(self.test_Scan_View[i:i + 1], test_hr_dir + '/Scan_View_%d.tif' % i)
            write3d(self.test_Synview[i:i + 1], test_mr_dir + '/SynView_%d.tif' % i)
            write3d(self.test_LFP[i:i + 1], test_lf_dir + '/LFP_%d.tif' % i)






    def _save_intermediate_ckpt(self, tag, sess):
        tag = ('epoch%d' % tag) if is_number(tag) else tag

        den_file_name = checkpoint_dir + '/denoise_net_{}.npz'.format(tag)
        sr_file_name = checkpoint_dir + '/SR_net_{}.npz'.format(tag)
        recon_file_name = checkpoint_dir + '/recon_net_{}.npz'.format(tag)

        tl.files.save_npz(self.denoise_net.all_params, name=den_file_name, sess=sess)
        tl.files.save_npz(self.SR_net.all_params, name=sr_file_name, sess=sess)
        tl.files.save_npz(self.Recon_net.all_params, name=recon_file_name, sess=sess)

        if 'epoch' in tag:
            if batch_size >= test_num:
                test_lr_batch = self.test_LFP[0:batch_size]
                denoise_view = self.sess.run(self.denoise_net.outputs, {self.plchdr_lf: test_lr_batch})
                SR_view = self.sess.run(self.SR_net.outputs, {self.plchdr_lf: test_lr_batch})
                Recon_stack = self.sess.run(self.Recon_net.outputs, {self.plchdr_lf: test_lr_batch})
                for i in range(test_num):
                    write3d(denoise_view[i:i + 1], test_saving_dir + ('denoise_{}_%d.tif' % (i)).format(tag))
                    write3d(SR_view[i:i + 1], test_saving_dir + ('SR_{}_%d.tif' % (i)).format(tag))
                    write3d(Recon_stack[i:i + 1], test_saving_dir + ('Recon_{}_%d.tif' % (i)).format(tag))
            else:
                for idx in range(0, test_num, batch_size):
                    if idx + batch_size <= test_num:

                        test_lr_batch = self.test_LFP[idx:idx + batch_size]
                        [denoise_view, SR_view, Recon_stack] = self.sess.run([self.denoise_net.outputs,self.SR_net.outputs,self.Recon_net.outputs], {self.plchdr_lf: test_lr_batch})
                        # Recon_stack = self.sess.run(self.Recon_net.outputs, {self.plchdr_lf: test_lr_batch})
                        # denoise_view = self.sess.run(self.denoise_net.outputs, {self.plchdr_lf: test_lr_batch})
                        for i in range(len(SR_view)):
                            write3d(denoise_view[i:i + 1],
                                    test_saving_dir + ('denoise_{}_%d.tif' % (i + idx * batch_size)).format(tag))
                            write3d(SR_view[i:i + 1],
                                    test_saving_dir + ('SR_{}_%d.tif' % (i + idx * batch_size)).format(tag))
                            write3d(Recon_stack[i:i + 1],
                                    test_saving_dir + ('Recon_{}_%d.tif' % (i + idx * batch_size)).format(tag))

    def _record_avg_test_loss(self, epoch, sess):
        if 'min_test_loss' not in dir(self):
            self.min_test_loss = 1e10
            self.best_epoch = 0
            self.test_loss_plt = []

        test_loss = 0
        test_data_num = len(self.test_LFP)
        print("")
        for idx in range(0, test_data_num, batch_size):
            if idx + batch_size <= test_data_num:
                test_lf_batch = self.test_LFP[idx: idx + batch_size]
                test_synview_batch = self.test_Synview[idx: idx + batch_size]
                test_scanview_batch = self.test_Scan_View[idx: idx + batch_size]
                test_target3d_batch = self.test_target3d[idx: idx + batch_size]

                feed_test = {self.plchdr_lf: test_lf_batch,
                             self.plchdr_SynView: test_synview_batch,
                             self.plchdr_Scan_View: test_scanview_batch,
                             self.plchdr_Target3D: test_target3d_batch
                             }

                test_loss_batch, losses_batch = sess.run([self.loss_test, self.losses], feed_test)
                loss_str = [name + ':' + str(value) for name, value in losses_batch.items() if 'loss' in name]
                test_loss += test_loss_batch
                print('\rvalidation  [% 2d/% 2d] loss = %.6f --%s ' % (idx, test_data_num, test_loss_batch, loss_str),
                      end='')
        test_loss /= (len(self.test_LFP) // batch_size)
        print('\navg = %.6f best = %.6f (@epoch%d)' % (test_loss, self.min_test_loss, self.best_epoch))
        self.test_loss_plt.append([epoch, test_loss])
        temp_file_name = plot_test_loss_dir + '/plot_test_loss.npy'
        np.save(temp_file_name, self.test_loss_plt)

        if (test_loss < self.min_test_loss):
            self.min_test_loss = test_loss
            self.best_epoch = epoch
            self._save_intermediate_ckpt(tag='best', sess=sess)
            # self._save_pb(sess)

    def _plot_test_loss(self):
        loss = np.asarray(self.test_loss_plt)
        plt.figure()
        plt.plot(loss[:, 0], loss[:, 1])
        plt.savefig(plot_test_loss_dir + '/test_loss.png', bbox_inches='tight')
        plt.show()

    def _traversal_through_ckpts(self, checkpoint_dir, epoch, label=None):
        ckpt_found = False
        filelist = os.listdir(checkpoint_dir)
        for file in filelist:
            if '.npz' in file and str(epoch) in file:
                if label is not None:
                    if label in file:
                        return file
                else:
                    return file
        return None

    def train(self, **kwargs):
        try:
            self._train(**kwargs)
        finally:
            self._plot_test_loss()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt', type=int, default=0, help='')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='')
    parser.add_argument('-cfg', '--config_path', type=str)
    args = parser.parse_args()
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import time
    import tensorflow as tf
    import tensorlayer as tl
    import numpy as np
    import matplotlib.pyplot as plt
    from model import *
    from dataset_joint import Dataset
    from utils import write3d, save_configs
    from config import configs_settings


    ###=================img pre ===========================###
    local_configs = configs_settings(args.config_path)

    img_size = local_configs.img_setting.img_size
    n_num = local_configs.img_setting.Nnum
    sr_factor = local_configs.img_setting.sr_factor
    n_slices = local_configs.img_setting.n_slices
    ReScale_factor = local_configs.img_setting.ReScale_factor
    base_size = img_size // n_num  # lateral size of lf_extra
    normalize_mode = local_configs.preprocess.normalize_mode
    sample_ratio = local_configs.TRAIN.sample_ratio
    test_num = 4

    ###=================training para ===========================###
    root_path = local_configs.root_path
    loading_pretrain_model = local_configs.Pretrain.loading_pretrain_model

    batch_size = local_configs.TRAIN.batch_size
    shuffle_for_epoch = local_configs.TRAIN.shuffle_for_epoch
    lr_init = local_configs.TRAIN.lr_init
    beta1 = local_configs.TRAIN.beta1
    n_epoch = local_configs.TRAIN.n_epoch
    lr_decay = local_configs.TRAIN.lr_decay
    decay_every = local_configs.TRAIN.decay_every

    ###=================dir ===========================###

    # PSF_path=local_configs.TRAIN.PSF_path
    label = local_configs.label
    test_saving_dir = os.path.join(root_path, 'DL', local_configs.TRAIN.test_saving_path)
    test_stack_dir = os.path.join(test_saving_dir, 'Target3D')
    test_hr_dir = os.path.join(test_saving_dir, 'Scan_View')
    test_mr_dir = os.path.join(test_saving_dir, 'Clean_View')
    test_lf_dir = os.path.join(test_saving_dir, 'BG_View')
    plot_test_loss_dir = os.path.join(test_saving_dir, 'test_loss_plt')

    checkpoint_dir = os.path.join(root_path, 'DL', local_configs.TRAIN.ckpt_dir)
    log_dir = os.path.join(root_path, 'DL', local_configs.TRAIN.log_dir)

    ckpt_saving_interval = local_configs.TRAIN.ckpt_saving_interval

    ###=================losses define ===========================###
    denoise_loss = local_configs.Loss.denoise_loss
    SR_loss = local_configs.Loss.SR_loss
    Recon_loss = local_configs.Loss.Recon_loss
    loss_ratio = local_configs.Loss.Ratio

    ## save_training_json
    local_pre_SRVCD_dict = local_configs.local_pre_SRVCD_dict
    preSRVCD_ckpt_save = checkpoint_dir + local_pre_SRVCD_dict['ckpt_save']


    args = parser.parse_args()
    to_disk=local_configs.TRAIN.to_Disk

    if to_disk:
        from dataset_tif_disk import Dataset_tif_disk

        Target3D_path=os.path.join(local_configs.img_setting.data_root_path,'sVol')
        Scan_view_path=os.path.join(local_configs.img_setting.data_root_path,'sLF')
        Synth_view_path=os.path.join(local_configs.img_setting.data_root_path,'cLF')
        LFP_path=os.path.join(local_configs.img_setting.data_root_path,'nLF')
        training_dataset = Dataset_tif_disk(Target3D_path,Scan_view_path,Synth_view_path,LFP_path,
                                   n_num,
                                   base_size,
                                   shuffle_for_epoch=shuffle_for_epoch,
                                   normalize_mode=normalize_mode,
                                   sample_ratio=sample_ratio,
                                   shuffle_all_data=local_configs.TRAIN.shuffle_all_data)
    else:
        if local_configs.img_setting.save_hdf5:
            base_path = os.path.join(local_configs.img_setting.data_root_path,'training_data.h5')
            data_str=['sVol','sLF','cLF','nLF']
        else:
            base_path = os.path.join(local_configs.img_setting.data_root_path)
            data_str=['sVol','sLF','cLF','nLF']
        training_dataset = Dataset(base_path=base_path,data_str=data_str,
                                   shuffle_for_epoch=shuffle_for_epoch,
                                   normalize_mode=normalize_mode,
                                   sample_ratio=sample_ratio,
                                   n_num=n_num,
                                   shuffle_all_data=local_configs.TRAIN.shuffle_all_data,
                                   save_hdf5=local_configs.img_setting.save_hdf5)

    trainer = Trainer(training_dataset)
    trainer.build_graph()
    trainer.train(begin_epoch=args.ckpt)
