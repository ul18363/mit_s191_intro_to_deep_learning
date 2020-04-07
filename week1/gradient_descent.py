# -*- coding: utf-8 -*-
import tensorflow as tf
"""
    This code needs a lot of rewriting in order to make a workable example.
    However it should be possible if I get some sample data X,Y
"""
#Define network
model = tf.keras.Sequential([...])

#Pick optimizar
optimizer = tf.keras.optimizer.SGD() #Define optimizer pick any!

while False: #Dont want to do the loop
    prediction=model(x)
    with tf.GradientTape() as tape:
        loss =compute_loss(y,prediction)
    grads= tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients( zip(grads,model.trainable_variables) ) 