# -*- coding: utf-8 -*-
# /usr/bin/python2

from __future__ import print_function

import os

from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from train import Graph
from utils import *
from data_load import load_data
from scipy.io.wavfile import write
from tqdm import tqdm
import time

def synthesize(text, output_path):

    start_synthesize_time = time.time()
    # Load data
    L = load_data("synthesize_one", text) # look at this load data function
    
    # Feed Forward
    ## mel
    Y = np.zeros((len(L), hp.max_T, hp.n_mels), np.float32)
    prev_max_attentions = np.zeros((len(L),), np.int32)
    for j in tqdm(range(hp.max_T)):
        _gs, _Y, _max_attentions, _alignments = \
            sess.run([g.global_step, g.Y, g.max_attentions, g.alignments],
                     {g.L: L,
                      g.mels: Y,
                      g.prev_max_attentions: prev_max_attentions})
        Y[:, j, :] = _Y[:, j, :]
        prev_max_attentions = _max_attentions[:, j]

    # Get magnitude
    print('Running Session')
    Z = sess.run(g.Z, {g.Y: Y})

    # Generate wav files    
    for i, mag in enumerate(Z):
        print("Working on file", i+1)
        wav = spectrogram2wav(mag)
        write(output_path, hp.sr, wav)
    print('Elapsed synthesize time: '+str(time.time() - start_synthesize_time))



# Load graph
model_load_time = time.time()
g = Graph(mode="synthesize"); print("Graph loaded")

sess = tf.Session()

sess.run(tf.global_variables_initializer())

# Restore parameters
var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Text2Mel')
saver1 = tf.train.Saver(var_list=var_list)
saver1.restore(sess, tf.train.latest_checkpoint(hp.logdir + "-1"))
print("Text2Mel Restored!")

var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'SSRN') + \
           tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'gs')
saver2 = tf.train.Saver(var_list=var_list)
saver2.restore(sess, tf.train.latest_checkpoint(hp.logdir + "-2"))
print("SSRN Restored!")
print('Elapsed model load time: '+str(time.time() - model_load_time))

        
#    synthesize(u"I do not believe that is correct", sess)    
#    synthesize(u"How much wood would a wood chuck chuck if a wood chuck would chuck wood", sess)    
#    print("Done")


