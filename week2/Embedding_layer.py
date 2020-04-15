# -*- coding: utf-8 -*-

# import tensorflow as tf
# from tensorflow import keras
from tensorflow.keras import layers,Sequential
import numpy as np
# import tensorflow_datasets as tfds
# tfds.disable_progress_bar()
# https://www.tensorflow.org/tutorials/text/word_embeddings#word_embeddings_2

"""
 An embedding is a dense vector of floating point values 
 (the length of the vector is a parameter you specify, it is either set empirically
 or optimized as an hyperparameter).
 
 The Embedding layer can be understood as a lookup table that maps from 
 integer indices (which stand for specific words) to dense vectors 
 (their embeddings).
"""

if __name__=='__main__':
    # embedding_layer = layers.Embedding(1000, 5)
    
    model = Sequential()
    
    #input_length          = 10 #Number of chars through at once to the model.
    #output_dimensionality = 64
    #batch_size            = 32 # 32 Examples will be submitted at a time
    #number_of_categories  = 1000
    
    model.add(layers.Embedding(1000, 64, input_length=10))
    wgs=model.weights[0].numpy()
    # the model will take as input an integer matrix of size (batch,
    # input_length).
    # the largest integer (i.e. word index) in the input should be no larger
    # than 999 (vocabulary size).
    # now model.output_shape == (None, 10, 64), where None is the batch
    # dimension.
    
    input_array_2 = np.random.randint(1000, size=(32, 5))
    input_array = np.random.randint(1000, size=(32, 10))
    embedded_input=model(input_array)
    embedded_input_np=embedded_input.numpy()
    # Shape=(32,10,64)->(
                         #Training sample,
                         #element,
                         #Embedded vector for embbedded input)
    
    
    
    if all(wgs[input_array[0][0]][:]==embedded_input_np[0][0][:]):
        print("""
              The first element of the first batch of the input has a number {i}
              The row {i} of the weight vector matches its embedded representation
              """.format(i=str(input_array[0][0])))
    
    embedded_input_2=model(input_array_2)