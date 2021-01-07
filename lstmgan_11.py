# conditional gan with cnn features flower data

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
import numpy as np
import model0_v1
import argparse
import pickle
from os.path import join
import h5py
from Utils import image_processing
import scipy.misc
import random
import json
from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream
import shutil
import time
import sys
from matplotlib import pyplot as plt


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--t_dim', type=int, default=256,
                        help='Text feature dimension')

    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch Size')

    parser.add_argument('--image_size', type=int, default=64,
                        help='Image Size a, a x a')

    parser.add_argument('--gf_dim', type=int, default=64,
                        help='Number of conv in the first layer gen.')

    parser.add_argument('--df_dim', type=int, default=64,
                        help='Number of conv in the first layer discr.')

    parser.add_argument('--gfc_dim', type=int, default=1024,
                        help='Dimension of gen untis for for fully connected layer 1024')

    parser.add_argument('--caption_vector_length', type=int, default=2400,
                        help='Caption Vector Length')

    parser.add_argument('--data_dir', type=str, default="Data",
                        help='Data Directory')

    parser.add_argument('--learning_rate', type=float, default=0.0002,
                        help='Learning Rate')

    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Momentum for Adam Update')

    parser.add_argument('--epochs', type=int, default=600,
                        help='Max number of epochs')

    parser.add_argument('--save_every', type=int, default=30,
                        help='Save Model/Samples every x iterations over batches')

    parser.add_argument('--resume_model', type=str, default=None,
                        help='Pre-Trained Model Path, to resume from')

    parser.add_argument('--data_set', type=str, default="flowers",
                        help='Dat set: MS-COCO, flowers')


    args = parser.parse_args()
    model_options = {
        't_dim': args.t_dim,
        'batch_size': args.batch_size,
        'image_size': args.image_size,
        'gf_dim': args.gf_dim,
        'df_dim': args.df_dim,
        'gfc_dim': args.gfc_dim,
        'caption_vector_length': args.caption_vector_length
    }

    gan = model0_v1.GAN(model_options)
    input_tensors, variables, loss, outputs, checks = gan.build_model()

    d_optim = tf.train.AdamOptimizer(args.learning_rate, beta1=args.beta1).minimize(loss['d_loss'],
                                                                                    var_list=variables['d_vars'])
    g_optim = tf.train.AdamOptimizer(args.learning_rate, beta1=args.beta1).minimize(loss['g_loss'],
                                                                                    var_list=variables['g_vars'] + variables['l_vars'])

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    saver = tf.train.Saver()

    nlen = 15
    nt = 2400
    n_updates = 0
    n_epochs = 0
    n_updates = 0
    n_examples = 0
    g_costs = []
    d_costs = []
    gl_costs = []
    ntrain = 6149
    nval = 1020

    dataPath = './flower_15.h5'
    train_set = H5PYDataset(dataPath, which_sets=('train',))
    train_set1 = H5PYDataset(dataPath, which_sets=('train',))
    val_set = H5PYDataset(dataPath, which_sets=('val',))
    val_set1 = H5PYDataset(dataPath, which_sets=('val',))

    tr_scheme = ShuffledScheme(examples=train_set.num_examples, batch_size=args.batch_size)
    tr_stream = DataStream(train_set, iteration_scheme=tr_scheme)

    tr_scheme1 = SequentialScheme(examples=train_set.num_examples, batch_size=args.batch_size)
    tr_stream1 = DataStream(train_set1, iteration_scheme=tr_scheme1)

    val_scheme = SequentialScheme(examples=val_set.num_examples, batch_size=args.batch_size)
    val_stream = DataStream(val_set, iteration_scheme=val_scheme)

    val_scheme1 = ShuffledScheme(examples=val_set1.num_examples, batch_size=args.batch_size)
    val_stream1 = DataStream(val_set1, iteration_scheme=val_scheme1)

    it_train = tr_stream.get_epoch_iterator()
    it_train1 = tr_stream1.get_epoch_iterator()
    it_val = val_stream.get_epoch_iterator()
    it_val1 = val_stream1.get_epoch_iterator()


    for epoch in range(args.epochs):
        n_batches = int(ntrain / args.batch_size)
        start_time = time.time()
        for batch in range(n_batches):
            try:
                annotationtr, image_featuretr, skip_vectortr, word_vectortr = it_train.next()
                image_featuretr = image_featuretr / np.float32(255)

                annotationtr1, image_featuretr1, skip_vectortr1, word_vectortr1 = it_train1.next()
                image_featuretr1 = image_featuretr1 / np.float32(255)
            except:
                it_train = tr_stream.get_epoch_iterator()
                annotationtr, image_featuretr, skip_vectortr, word_vectortr = it_train.next()
                image_featuretr = image_featuretr / np.float32(255)

                it_train1 = tr_stream1.get_epoch_iterator()
                annotationtr1, image_featuretr1, skip_vectortr1, word_vectortr1 = it_train1.next()
                image_featuretr1 = image_featuretr1 / np.float32(255)


            if annotationtr.shape[0] != args.batch_size:
                it_train = tr_stream.get_epoch_iterator()
                annotationtr, image_featuretr, skip_vectortr, word_vectortr = it_train.next()
                image_featuretr = image_featuretr / np.float32(255)

                it_train1 = tr_stream1.get_epoch_iterator()
                annotationtr1, image_featuretr1, skip_vectortr1, word_vectortr1 = it_train1.next()
                image_featuretr1 = image_featuretr1 / np.float32(255)

            image_featuretr = np.reshape(image_featuretr, [image_featuretr.shape[0], -1])
            image_featuretr = np.reshape(image_featuretr, (image_featuretr.shape[0], 64, 64, 3))
            #image_featuretr = np.transpose(image_featuretr, [0, 3, 1, 2])
            image_featuretr_new = np.empty((args.batch_size, nlen, 64, 64, 3))
            for m in range(args.batch_size):
                for mm in range(nlen):
                    image_featuretr_new[m][mm] = image_featuretr[m]
            image_featuretr_new = np.asarray(image_featuretr_new, dtype='float32')
            imb0 = image_featuretr
            imb = image_featuretr  # batchsize*3*64*64

            image_featuretr1 = np.reshape(image_featuretr1, [image_featuretr1.shape[0], -1])
            image_featuretr1 = np.reshape(image_featuretr1, (image_featuretr1.shape[0], 64, 64, 3))
            #image_featuretr1 = np.transpose(image_featuretr1, [0, 3, 1, 2])
            image_featuretr1_new = np.empty((args.batch_size, nlen, 64, 64, 3))
            for m in range(args.batch_size):
                for mm in range(nlen):
                    image_featuretr1_new[m][mm] = image_featuretr1[m]
            image_featuretr1_new = np.asarray(image_featuretr1_new, dtype='float32')
            imb1 = image_featuretr1  # batchsize*3*64*64

            y_sum = np.sum(word_vectortr, axis=2)
            y_len = np.empty((args.batch_size,), dtype='int32')
            for j in range(args.batch_size):
                y_len[j] = np.count_nonzero(y_sum[j]) - 1


            # con = skip_vectortr[:, 0:2400]  # batchsize*2400
            # con = np.reshape(con, (args.batch_size, nt))  # batchsize*2400
            #zmb = word_vectortr  # batchsize*length*400
            con = word_vectortr
            z_noise = np.random.uniform(-1, 1, [args.batch_size, 100])

            # DISCR UPDATE
            check_ts = [checks['d_loss1'], checks['d_loss2'], checks['d_loss3']]
            _, d_loss, gen, d1, d2, d3 = sess.run([d_optim, loss['d_loss'], outputs['generator']] + check_ts,
                                                  feed_dict={
                                                      input_tensors['t_real_image']: imb,
                                                      input_tensors['t_wrong_image']: imb1,
                                                      input_tensors['t_real_caption']: con,
                                                      #input_tensors['t_z']: z_noise,
                                                  })

            print "d1", d1
            print "d2", d2
            print "d3", d3
            print "D", d_loss

            # GEN UPDATE
            _, g_loss, gen = sess.run([g_optim, loss['g_loss'], outputs['generator']],
                                      feed_dict={
                                          input_tensors['t_real_image']: imb,
                                          input_tensors['t_wrong_image']: imb1,
                                          input_tensors['t_real_caption']: con,
                                          #input_tensors['t_z']: z_noise,
                                      })

            # GEN UPDATE TWICE, to make sure d_loss does not go to 0
            _, g_loss, gen = sess.run([g_optim, loss['g_loss'], outputs['generator']],
                                      feed_dict={
                                          input_tensors['t_real_image']: imb,
                                          input_tensors['t_wrong_image']: imb1,
                                          input_tensors['t_real_caption']: con,
                                          #input_tensors['t_z']: z_noise,
                                      })

            print 'batch:%.0f g_cost_img: %.4f d_cost_img:%.4f' % (batch, float(g_loss), float(d_loss))

            n_updates += 1
            n_examples += len(imb)

        t = time.time()
        print 'epoch:%.0f  time: %d g_cost_img: %.4f d_cost_img:%.4f' \
              % (epoch + 1, int((t - start_time) / 0.01) / 100.0, float(g_loss), float(d_loss))

        n_epochs += 1
        # if n_epochs > 50:
        #    progress = float(epoch) / num_epochs
        #    eta.set_value(lasagne.utils.floatX(initial_eta * 2 * (1 - progress)))

        sys.stdout.flush()

        annotationtr1 = annotationtr[0:16]
        for u in range(16):
            file = open("./pdf_11/train/txt/%s.txt" % str(n_epochs), "a")
            xxxx = annotationtr1[u][0]
            file.write(str(u + 1) + ' ' + xxxx + '\n')
            file.close()

        # generative image
        reconst_img = gen  # length*batchsize*(64*64*3)
        # reconst_img = np.array(reconst_img)
        # reconst_img = np.reshape(reconst_img, (batchsize, nlen, 3, 64, 64))
        # reconst_img = reconst_img.transpose(0, 2, 3, 1)  # batchsize*length*(64*64*3)

        # figs, axes = plt.subplots(4, 4, figsize=(16, 16))
        # for i in range(4):
        #     for j in range(4):
        #         axes[i, j].imshow(reconst_img[i * 4 + j])
        #         axes[i, j].set_xticks([])
        #         axes[i, j].set_yticks([])
        #         axes[i, j].axis('off')
        # figs.savefig('./pdf_11/train/gen/' + str(n_epochs) + '.png')
        # plt.close()

        figs, axes = plt.subplots(4, 4, figsize=(16, 16))
        for i in range(4):
            for j in range(4):
                axes[i, j].imshow(reconst_img[i * 4 + j])
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
                axes[i, j].axis('off')
        figs.savefig('./pdf_11/train/gen/' + str(n_epochs) + '.png')
        plt.close()


        reconst_img2 = imb0
        #reconst_img2 = reconst_img2.transpose(0, 2, 3, 1)
        figs, axes = plt.subplots(4, 4, figsize=(16, 16))
        for i in range(4):
            for j in range(4):
                axes[i, j].imshow(reconst_img2[i * 4 + j])
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
                axes[i, j].axis('off')
        figs.savefig('./pdf_11/train/real/' + str(n_epochs) + '.png')
        plt.close()

        # # Validation model
        # cost_val = _val(imbva, imbva1, conva)
        #
        # print 'validation epoch:%.0fd_cost_img:%.4f' % (epoch, float(cost_val))
        #
        # sys.stdout.flush()
        # annotationva1 = annotationva[0:16]
        #
        # for u in range(16):
        #     file = open("./pdf_11/val/txt/%s.txt" % str(n_epochs), "a")
        #     xxxx = annotationva1[u][0]
        #     file.write(str(u + 1) + ' ' + xxxx + '\n')
        #     file.close()
        #
        # reconst_img = _gen(conva)  # length*batchsize*(3*64*64)
        # # reconst_img = np.array(reconst_img)
        # # reconst_img = np.reshape(reconst_img, (batchsize, nlen, 3, 64, 64))
        # reconst_img = reconst_img.transpose(0, 2, 3, 1)  # batchsize*length*(64*64*3)
        #
        # figs, axes = plt.subplots(4, 4, figsize=(16, 16))
        # for i in range(4):
        #     for j in range(4):
        #         axes[i, j].imshow(reconst_img[i * 4 + j])
        #         axes[i, j].set_xticks([])
        #         axes[i, j].set_yticks([])
        #         axes[i, j].axis('off')
        # figs.savefig('./pdf_11/val/gen/' + str(n_epochs) + '.png')
        # plt.close()
        #
        # reconst_img1 = imbva0
        # reconst_img1 = reconst_img1.transpose(0, 2, 3, 1)
        # figs, axes = plt.subplots(4, 4, figsize=(16, 16))
        # for i in range(4):
        #     for j in range(4):
        #         axes[i, j].imshow(reconst_img1[i * 4 + j])
        #         axes[i, j].set_xticks([])
        #         axes[i, j].set_yticks([])
        #         axes[i, j].axis('off')
        # figs.savefig('./pdf_11/val/real/' + str(n_epochs) + '.png')
        # plt.close()

        if epoch % 50 == 0:
            saver.save(sess, "./pdf_11/model/model_after_{}_epoch_{}.ckpt".format(args.data_set, epoch))



def load_training_data(data_dir, data_set):
	h = h5py.File(join(data_dir, 'flower_15.h5'))
	flower_captions = {}
	for ds in h.iteritems():
		flower_captions[ds[0]] = np.array(ds[1])
	image_list = [key for key in flower_captions]
	image_list.sort()

	img_75 = int(len(image_list)*0.75)
	training_image_list = image_list[0:img_75]
	random.shuffle(training_image_list)

	return {
		'image_list' : training_image_list,
		'captions' : flower_captions,
		'data_length' : len(training_image_list)
		}


def save_for_vis(data_dir, real_images, generated_images, image_files):

	shutil.rmtree( join(data_dir, 'samples') )
	os.makedirs( join(data_dir, 'samples') )

	for i in range(0, real_images.shape[0]):
		real_image_255 = np.zeros( (64,64,3), dtype=np.uint8)
		real_images_255 = (real_images[i,:,:,:])
		scipy.misc.imsave( join(data_dir, 'samples/{}_{}.jpg'.format(i, image_files[i].split('/')[-1] )) , real_images_255)

		fake_image_255 = np.zeros( (64,64,3), dtype=np.uint8)
		fake_images_255 = (generated_images[i,:,:,:])
		scipy.misc.imsave(join(data_dir, 'samples/fake_image_{}.jpg'.format(i)), fake_images_255)


def get_training_batch(batch_no, batch_size, image_size, z_dim,
	caption_vector_length, split, data_dir, data_set, loaded_data = None):
	if data_set == 'flowers':
		real_images = np.zeros((batch_size, 64, 64, 3))
		wrong_images = np.zeros((batch_size, 64, 64, 3))
		captions = np.zeros((batch_size, caption_vector_length))

		cnt = 0
		image_files = []
		for i in range(batch_no * batch_size, (batch_no+1) * batch_size):
			idx = i % len(loaded_data['image_list'])
			image_file =  join(data_dir, 'flowers/jpg/'+loaded_data['image_list'][idx])
			image_array = image_processing.load_image_array(image_file, image_size)
			real_images[cnt,:,:,:] = image_array

			# Improve this selection of wrong image
			wrong_image_id = random.randint(0,len(loaded_data['image_list'])-1)
			wrong_image_file =  join(data_dir, 'flowers/jpg/'+loaded_data['image_list'][wrong_image_id])
			wrong_image_array = image_processing.load_image_array(wrong_image_file, image_size)
			wrong_images[cnt, :,:,:] = wrong_image_array

			random_caption = random.randint(0,4)
			captions[cnt,:] = loaded_data['captions'][ loaded_data['image_list'][idx] ][ random_caption ][0:caption_vector_length]
			image_files.append( image_file )
			cnt += 1

		z_noise = np.random.uniform(-1, 1, [batch_size, z_dim])
		return real_images, wrong_images, captions, z_noise, image_files

if __name__ == '__main__':
    main()