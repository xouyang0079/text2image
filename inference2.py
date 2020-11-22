import pickle
import numpy as np
from scipy.stats.stats import pearsonr
#from pyemd import emd
import cv2

path = './test_11f1.pkl'

with open(path, 'rb') as f:
    data = pickle.load(f)

ours_vgg_2 = data[0]
real_vgg_2 = data[1]

# with open(path1, 'rb') as f:
#     data = pickle.load(f)
#
# cgan_vgg_2 = data[0]
# cgan_vgg_r2 = data[1]

def computeEuDist(ours, real):
    diff = ours-real
    bsize = diff.shape[0]
    diff = np.reshape(diff, [bsize, -1])
    diff = np.power(diff, 2)
    diff = np.sum(diff, axis = 1)
    diff = np.sqrt(diff)
    diff = np.mean(diff)
    return diff

def correlation(ours, real):
    bsize = ours.shape[0]
    ours = np.reshape(ours, [bsize, -1])
    real = np.reshape(real, [bsize, -1])
    corr = 0
    for i in range(bsize):
        c, _ = pearsonr(ours[i], real[i])
        corr = corr + c
    return corr/bsize

### Euclidean distance
ours_euDist_vgg_2 = computeEuDist(ours_vgg_2, real_vgg_2)
ours_corr_vgg_2 = correlation(ours_vgg_2, real_vgg_2)

print "ours_euDist_vgg_2: %f"% ours_euDist_vgg_2
print "ours_corr_vgg_2: %f"% ours_corr_vgg_2

# cgan_euDist_vgg_2 = computeEuDist(cgan_vgg_2, real_vgg_2)
# cgan_corr_vgg_2 = correlation(cgan_vgg_2, real_vgg_2)
#
# cgan_euDist_vgg_r2 = computeEuDist(cgan_vgg_r2, real_vgg_r2)
# cgan_corr_vgg_r2 = correlation(cgan_vgg_r2, real_vgg_r2)