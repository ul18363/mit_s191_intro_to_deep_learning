# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 01:33:53 2020

@author: Docker
"""



"""
    Regularization technique #1: DropOut
"""
import tensorflow as tf

#During training randomly set some activations to 0
# Makes the network not to rely on any particular neuron (Forces redundancy in the network)
# On every iteration 50% of the memory of previous data is going to be wiped out
# Implements multiple channels for the representation of a feature
tf.keras.layers.Dropout(p=0.5)



"""
    Regularization technique #2: EarlyStopping

    Stop before having the chance to overfit. Identify the inflection point between the error
    of the testing error and use set of weights of the minimum 
"""

