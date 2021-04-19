import numpy as np
import tensorflow as tf
import h5py
import os
import time

def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

annot_dir = 'data/MPII/annot'
img_dir = 'data/MPII/images'

assert os.path.exists(img_dir)
mpii = None

import cv2

class MPII:
    def __init__(self, annot):
        print('loading data:', annot)
        tic = time.time()

        ds_f = h5py.File(os.path.join(annot_dir, annot), 'r')
        
        self.center = ds_f['center'][()]
        self.scale = ds_f['scale'][()]
        self.part = ds_f['part'][()]
        self.visible = ds_f['visible'][()]
        self.normalize = ds_f['normalize'][()]
        self.imgname = [None] * len(self.center)
        for i in range(len(self.center)):
            self.imgname[i] = ds_f['imgname'][i].decode('UTF-8')
        self.part_reference()
        print('Done (t={:0.2f}s)'.format(time.time()- tic))
        
    def getAnnots(self, idx):
        '''
        returns h5 file for train or val set
        '''
        return self.imgname[idx], self.part[idx], self.visible[idx], self.center[idx], self.scale[idx], self.normalize[idx]
    
    def getLength(self):
        return len(self.center)

    def get_img(self, idx):
        imgname, __, __, __, __, __ = self.getAnnots(idx)
        path = os.path.join(img_dir, imgname)
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        return img

    def get_path(self, idx):
        imgname, __, __, __, __, __ = self.getAnnots(idx)
        path = os.path.join(img_dir, imgname)
        return path
        
    def get_kps(self, idx):
        __, part, visible, __, __, __ = self.getAnnots(idx)
        kp2 = np.insert(part, 2, visible, axis=1)
        kps = np.zeros((1, 16, 3))
        kps[0] = kp2
        return kps

    def get_normalized(self, idx):
        __, __, __, __, __, n = self.getAnnots(idx)
        return n

    def get_center(self, idx):
        __, __, __, c, __, __ = self.getAnnots(idx)
        return c
        
    def get_scale(self, idx):
        __, __, __, __, s, __ = self.getAnnots(idx)
        return s

    def part_reference(self):
        self.parts = {'mpii':['rank', 'rkne', 'rhip',
                 'lhip', 'lkne', 'lank',
                 'pelv', 'thrx', 'neck', 'head',
                 'rwri', 'relb', 'rsho',
                 'lsho', 'lelb', 'lwri']}

        self.flipped_parts = {'mpii':[5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 15, 14, 13, 12, 11, 10]}

        self.part_pairs = {'mpii':[[0, 5], [1, 4], [2, 3], [6], [7], [8], [9], [10, 15], [11, 14], [12, 13]]}

        self.pair_names = {'mpii':['ankle', 'knee', 'hip', 'pelvis', 'thorax', 'neck', 'head', 'wrist', 'elbow', 'shoulder']}