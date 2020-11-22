# Text2image

## Paper
Skip Thought Vectors:  
        paper: https://arxiv.org/pdf/1506.06726.pdf  
        source: https://github.com/paarthneekhara/text-to-image  
    
Generating Image Sequence from Description with LSTM Conditional GAN:  
        paper: https://arxiv.org/pdf/1806.03027.pdf  
    
## Requirements  
* Python 2.7.6  
* Tensorflow 1.5.0   
* Lasagne
* h5py  
* fuel  
* compiler  
* gensim
* Theano : for skip thought vectors  
* scikit-learn : for skip thought vectors  
* NLTK : for skip thought vectors  

## Dataset  
* Oxford-102 flower dataset: http://www.robots.ox.ac.uk/~vgg/data/flowers/102/  
* Caltech-UCSD Birds-200-2011 dataset: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html  

## Usage  
* __Data process__  
1. Extract skip-thought vectors: https://github.com/paarthneekhara/text-to-image  
2. Generate h5 file for training and testing: `python h5flowerlast.py`

* __Training__  
`python lstmgan_11.py`  

* __Generating images from captions__  
`python test_1_1.py`  

* __Inference for MSE and SSIM__  
`python inference1.py`  

* __Extract vgg features__  
`python vgg_feature.py`  

* __Inference for Euclidean distance and Correlation__  
`python inference2.py`





        
