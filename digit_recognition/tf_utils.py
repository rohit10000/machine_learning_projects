import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops

def one_hot_matrix(labels, C):
    C = tf.constant(C, name = "C")
    
    one_hot_matrix = tf.one_hot(indices = labels , depth = C, axis = 0 )
    
    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()
        
    return one_hot