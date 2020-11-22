# lstm conditional gan without mask
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import numpy as np
import model_1
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
import cv2
from PIL import Image



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

    parser.add_argument('--data_dir', type=str, default="./",
                       help='Data Directory')

    parser.add_argument('--model_path', type=str, default='./pdf_11/model/model_after_flowers_epoch_550.ckpt',
                       help='Trained Model Path')

    parser.add_argument('--n_images', type=int, default=16,
                       help='Number of Images per Caption')



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

    gan = model_1.GAN(model_options)
    _, _, _, _, _ = gan.build_model()
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, args.model_path)

    input_tensors, outputs = gan.build_generator()

    nlen = 15
    nt = 2400
    n_updates = 0
    n_epochs = 0
    n_updates = 0
    n_examples = 0
    g_costs = []
    d_costs = []
    gl_costs = []
    ntest = 1020


    #path_last = "/home/xu/PycharmProjects/tensorflowlstmgan/pdf_1/test2/gen"


    dataPath = './flower_15.h5'
    test_set = H5PYDataset(dataPath, which_sets=('test',))

    test_scheme = SequentialScheme(examples=test_set.num_examples, batch_size=args.batch_size)
    test_stream = DataStream(test_set, iteration_scheme=test_scheme)
    it_test = test_stream.get_epoch_iterator()


    for epoch in range(16):
        try:
            annotationtr, image_featuretr, skip_vectortr, word_vectortr = it_test.next()
            image_featuretr = image_featuretr / np.float32(255)

        except:
            it_test = test_stream.get_epoch_iterator()
            annotationtr, image_featuretr, skip_vectortr, word_vectortr = it_test.next()
            image_featuretr = image_featuretr / np.float32(255)

        image_featuretr = np.reshape(image_featuretr, [image_featuretr.shape[0], -1])
        image_featuretr = np.reshape(image_featuretr, (image_featuretr.shape[0], 64, 64, 3))
        # image_featuretr = np.transpose(image_featuretr, [0, 3, 1, 2])
        image_featuretr_new = np.empty((args.batch_size, nlen, 64, 64, 3))
        for m in range(args.batch_size):
            for mm in range(nlen):
                image_featuretr_new[m][mm] = image_featuretr[m]
        image_featuretr_new = np.asarray(image_featuretr_new, dtype='float32')
        imb0 = image_featuretr
        imb = image_featuretr  # batchsize*length*3*64*64


        y_sum = np.sum(word_vectortr, axis=2)
        y_len = np.empty((args.batch_size,), dtype='int32')
        for j in range(args.batch_size):
            y_len[j] = np.count_nonzero(y_sum[j]) - 1

        mask = np.empty((args.batch_size, 15, 400), dtype='float32')
        for i in range(args.batch_size):
            for j in range(15):
                if j < (y_len[i] + 1):
                    mask[i][j] = 1.0
                else:
                    mask[i][j] = 0.0

        con = skip_vectortr[:, 0:2400]  # batchsize*2400
        con = np.reshape(con, (args.batch_size, nt))  # batchsize*2400
        zmb = word_vectortr  # batchsize*length*400


        # GEN UPDATE TWICE, to make sure d_loss does not go to 0
        [gen_image] = sess.run([outputs['generator']],
                                         feed_dict={input_tensors['t_z']: zmb,
                                                    input_tensors['mask']: mask})


        n_updates += 1
        n_examples += len(imb)
        n_epochs += 1
        # if n_epochs > 50:
        #    progress = float(epoch) / num_epochs
        #    eta.set_value(lasagne.utils.floatX(initial_eta * 2 * (1 - progress)))

        sys.stdout.flush()

        annotationtr1 = annotationtr
        reconst_img2 = imb0
        for u in range(64):
            file = open("./pdf_11/test/txt/%s.txt" % str(n_epochs), "a")
            xxxx = annotationtr1[u][0]
            file.write(str(u + 1) + ' ' + xxxx + '\n')
            file.close()
            scipy.misc.imsave("./pdf_11/test/real/%i_%i.png" % (epoch, u), (reconst_img2[u] * 255.0).astype(int))

        # generative image
        reconst_img = gen_image  # length*batchsize*(64*64*3)
        print reconst_img.shape
        for m in range(64):
            gen_img = reconst_img[m][14]
            scipy.misc.imsave("./pdf_11/test/gen/%i_%i.png" % (epoch, m), (gen_img * 255.0).astype(int))

        print "%i finished"%(epoch)



if __name__ == '__main__':
    main()