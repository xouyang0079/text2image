import scipy.io as sio
import random
import h5py
import os, os.path
from fuel.datasets.hdf5 import H5PYDataset
from fuel.schemes import ShuffledScheme, SequentialScheme
from fuel.streams import DataStream
import PIL
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import glob
import re
import gensim
import copy
import pprint, pickle
from compiler.ast import flatten

# load annotation
def Loadfile(file):
    f = open(file)
    result = list()
    while 1:
        line = f.readline().strip('\n')
        result.append(line)
        if not line:
            break
    f.close()
    return result

# remove punctuation
def go_alert_to_space(list_a):
    mod_list = list()
    for i in range(len(list_a)):
        mod_list.append(re.sub('[,?.*/+-:;@#$%&|()]', '', list_a[i]))
    return mod_list

# substitute 'a' by 'one'
def suba(mod_list):
    for i in range(len(mod_list)):
        for j in range(len(mod_list[i])):
            if mod_list[i][j] == 'a':
                mod_list[i][j] = 'one'
            else:
                continue
    return mod_list

# transport to
def doubleBody(modlist):
    double_body = list()
    for i in range(len(modlist)):
        double_body.append(modlist[i].split())
    return double_body

# filter words which are not in wiki
def filterwiki(double_body):
    ret = list()
    def fun1(s): return s if s in model else None
    for i in range(len(double_body)):
            ret.append(filter(fun1, double_body[i]))
    return ret

# maxwords length
def subBody(double_body):
    for i in range(len(double_body)):
        if len(double_body[i]) > 25:
            for r in range(0,5):
                j = i/5*5+r
                if len(double_body[j]) > 25:
                    continue
                else:
                    double_body[i] = double_body[j]
    return double_body


if __name__ == "__main__":

    f = h5py.File('./flower_15.h5', 'w')

    image_feature = f.create_dataset('image_feature', shape=(1,3,64,64),
                                maxshape=(None,3,64,64), chunks=True, dtype='float32')
    word_vector = f.create_dataset('word_vector', shape=(1, 25, 400),
                                   maxshape=(None, 25, 400), chunks=True, dtype='float32')
    string_dt = h5py.special_dtype(vlen=str)
    annotation = f.create_dataset('annotation',shape=(1, 1),
                                maxshape=(None, 1), chunks=True, dtype=string_dt)

    model = gensim.models.KeyedVectors.load_word2vec_format("wiki.en.text.vector", binary=False)
    print 'load model finish.'
# extract word2vec
    path = "../Data/oxflower/cvpr2016_flowers/text_c10"
    imgfile = "../Data/oxflower/image64"

    image = np.empty((8189,3,64,64),dtype="float32")
    idx = 0
    annot = []
    filepath = os.path.join(imgfile, '*.jpg')
    for i in range(102):
        filepath1 = os.path.join(imgfile, 'class_%05d/*.jpg' % (i + 1))
        filepath2 = os.path.join(path, 'class_%05d/*.txt' % (i + 1))
        for files1,files2 in zip(sorted(glob.glob(filepath1)), sorted(glob.glob(filepath2))):
            for j in range(10):
                content = Loadfile(files2)[j]
                skip_c = (re.sub('[,?.*/+-:;@#$%&|()]', '', content)).split()
                if len(skip_c) < 16:
                    annot.append(content)
                    for j in range(len(filter)):
                        ss = ' '.join(word for word in content.split()[0:j + 1])
                    img = Image.open(files1)
                    img = np.asarray(img,dtype="float32")
                    img = np.reshape(img, (3,64,64))
                    image[idx,:,:,:] = img
                    idx += 1

        print "catagory %d load" % (i + 1)

    # skip_f = np.zeros((8189, 25, 2400))
    # for i in range(8189):
    #     for j in range(25):
    #         if j < len(skip_content[i]):
    #             skip_out = skipthoughts.encode(model, [skip_content[i][j]])
    #             skip_f[i][j] = skip_out[0][0:2400]
    #     print "fininsh %i"%i

    annot1 = flatten(annot)
    modlist = go_alert_to_space(annot1)
    double_body = doubleBody(modlist)
    sublist = suba(double_body)
    ret = filterwiki(sublist)

    # reduced matrix
    #final_body = subBody(ret)
    reducedwords = np.array(ret)

    reducedwords1 = np.zeros((8189, 25, 400))
    for i in range(0, 8189):
        for j in range(len(reducedwords[i])):
            reducedwords1[i][j] = model[reducedwords[i][j]]
    reduced = reducedwords1

    print "feature generated finished"

    body = np.asarray(annot, dtype=object)
    body = np.reshape(body, (8189,1))

# shuffle
    index = [i for i in range(8189)]
    random.shuffle(index)
    imagef = image[index]
    caption_vectorsf = reduced[index]
    #caption_vectorsf = skip_f[index]
    bodyf = body[index]

    # imagef = image
    # caption_vectorsf = reduced
    # bodyf = body

    for i in range(len(imagef)):
        image_feature[i:] = imagef[i]

        if i < len(imagef) - 1:
            image_feature.resize(i + 2, axis=0)

    for i in range(len(caption_vectorsf)):
        word_vector[i:] = caption_vectorsf[i]

        if i<len(caption_vectorsf)-1:
            word_vector.resize(i+2, axis=0)

    for i in range(len(bodyf)):
        annotation[i:] = bodyf[i]

        if i<len(bodyf)-1:
            annotation.resize(i+2, axis=0)

    bound0 = 6149
    bound1 = 7169
    bound2 = 8189

    split_dict = {'train': {'annotation': (0, bound0), 'image_feature': (0, bound0), 'word_vector': (0, bound0)},
                  'val': {'annotation': (bound0, bound1), 'image_feature': (bound0, bound1), 'word_vector': (bound0, bound1)},
                  'test': {'annotation': (bound1, bound2), 'image_feature': (bound1, bound2), 'word_vector': (bound1, bound2)}
                  }

    f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
    f.flush()
    f.close()
