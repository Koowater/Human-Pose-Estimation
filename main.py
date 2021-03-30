from model import HPE, HeatmapLoss, lr_scheduler
from data.dp import Dataset
import utils
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os
import cv2
import math

model = HPE()
# checkpoint

checkpoint_path = "data/checkpoint/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq='epoch')

model.save_weights(checkpoint_path.format(epoch=0))

config = {
    'data_provider': 'data.MPII.dp',
    'network': 'models.posenet.PoseNet',
    'inference': {
        'nstack': 8,
        'inp_dim': 256,
        'oup_dim': 16,
        'num_parts': 16,
        'increase': 0,
        'keys': ['imgs'],
        'num_eval': 2958, ## number of val examples used. entire set is 2958
        'train_num_eval': 300, ## number of train examples tested at test time
    },

    'train': {
        'batchsize': 16,
        'input_res': 256,
        'output_res': 64,
        'train_iters': 1000,
        'valid_iters': 10,
        'learning_rate': 1e-3,
        'max_num_people' : 1,
        'loss': [
            ['combined_hm_loss', 1],
        ],
        'decay_iters': 100000,
        'decay_lr': 2e-4,
        'num_workers': 2,
        'use_data_loader': True,
    },
}

t_ds = Dataset(config, 'train.h5')
v_ds = Dataset(config, 'valid.h5')
train = tf.data.Dataset.from_generator(t_ds.gen,
                                         output_types=(np.float32, np.float32),
                                         output_shapes=([None, 256, 256, 3], [None, 64, 64, 16]),
                                         args=())
valid = tf.data.Dataset.from_generator(v_ds.gen,
                                         output_types=(np.float32, np.float32),
                                         output_shapes=([None, 256, 256, 3], [None, 64, 64, 16]),
                                         args=())

lr = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
opt = tf.keras.optimizers.RMSprop(
    learning_rate=1e-04, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False,
    name='RMSprop')
model.compile(
    optimizer=opt, loss=HeatmapLoss, metrics=None, loss_weights=None,
    sample_weight_mode=None, weighted_metrics=None, target_tensors=None,
    distribute=None)
model.build((None, 256, 256, 3))

hist = model.fit(train, validation_data=valid, batch_size=16, callbacks=[lr, cp_callback], epochs=50)