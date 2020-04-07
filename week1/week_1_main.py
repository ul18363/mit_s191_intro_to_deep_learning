# -*- coding: utf-8 -*-


import sample_generator as sample_generator
import numpy as np
import tensorflow as tf
from os import system
from IPython import get_ipython
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#import pygame, sys
#import pygame.locals


def loss(tf_target,y_pred):
    return tf.math.sqrt(tf.reduce_sum(tf.math.square(tf_target-y_pred,2)))/tf_target.shape.num_elements()


#pygame.init()
   
sg=sample_generator.sample_generator()
n_features=2
n_samples=200
x_min=0
x_max=10
iterations=10
plotting=True
fancy_printing=False#True
make_educated_guess=False
sharpness=1
refresh_window=100
new_sample=sg.generate_sample(n_features,n_samples,x_min,x_max)

params= np.array([-4,-1,2])
target=sg.linearly_separated_target(new_sample,params=params)
#sg.visualize_sample(new_sample,target=target,params=params,label_data=True)
#model = tf.keras.Sequential([
#            tf.keras.layers.Dense(3,input_shape=(n_features,)), # 2 Intermediate layers
#            tf.keras.layers.Dense(1)  # 1 Output layer  
#        ])
model = tf.keras.Sequential([
            tf.keras.layers.Dense(1,input_shape=(n_features,),
                                  activation="sigmoid"),
#            tf.keras.layers.Dense(1,activation="sigmoid")
        ])

tf_target=tf.Variable(target)
tf_target=tf.cast(tf_target,tf.float32)
tf_target=tf.reshape(tf_target,(n_samples,1))

if make_educated_guess:
    manual_weights=[np.array([[params[1]*sharpness],
           [params[2]*sharpness]], dtype=np.float32), np.array([params[0]*sharpness], dtype=np.float32)]
    model.set_weights(manual_weights)
x=new_sample[:,0]
y=new_sample[:,1]
b=params[0]/params[2]
m=params[1]/params[2]
xl=[x_min,x_max]
yl=[-b- m*xl[0],-b-m*xl[1]]

if plotting:
    fig, axs = plt.subplots(1, 2)
    c=[x[0] for x in tf_target.numpy()]
    c=np.round(c,decimals=3)
    axs[0].set_title('Real')
    axs[0].scatter(x, y, c=c,cmap='rainbow')
    axs[0].plot(xl,yl)
    axs[0].axis(
                    (x_min,x_max,
                     x_min,x_max)
                    )
lr=0.05
epoch=0
while True:
    #Generate new samples for each iteration
#    new_sample=sg.generate_sample(n_features,n_samples,x_min,x_max)
#    target=sg.linearly_separated_target(new_sample,params=params)
#    tf_target=tf.Variable(target)
#    tf_target=tf.cast(tf_target,tf.float32)
    epoch=epoch+1
    with tf.GradientTape() as t:
        current_loss=loss(tf_target, model(new_sample))
        with t.stop_recording():
            # The gradient computation below is not traced, saving memory.
            grads = t.gradient(current_loss, model.variables)
    current_weights=model.get_weights()
    gradient=[x.numpy() for x in grads]
    new_weights=np.add(
                        current_weights,
                        np.multiply(gradient,-lr)
                        )
#   y_pred=np.round(model(new_sample).numpy(),decimals=3)
    model.set_weights(new_weights)
    if fancy_printing:
        system('cls')
        
#        get_ipython().magic('cls')
#        get_ipython().magic('reset -f')
#        print()
        print('Loss: '+str(current_loss.numpy()))
        c_wg=[str(float(x))[:5] for y in current_weights for x in y]
        print('Weights:'+'|'.join(c_wg))
        c_gr=[str(float(x))[:5] for y in gradient for x in y]
        print('Weights:'+'|'.join(c_gr))
        
        print('Epoch: '+str(epoch))
        
    if plotting and (epoch%refresh_window==0):
        axs[1].set_title('Pred[lr: '+str(lr)+'] loss:'+str(current_loss.numpy()))
        c=[x[0] for x in model(new_sample).numpy()]
        c=np.round(c,decimals=3)
        axs[1].scatter(x, y, c=c,cmap='rainbow')
        axs[1].plot(xl,yl)
        axs[1].axis(
                        (x_min,x_max,
                         x_min,x_max)
                        )
        fig.canvas.draw()
        plt.pause(0.005)
        a=input("Press Enter to Continue...")
        if not a:
            continue
        elif a=='q':
            break
        elif a=='d':
            lr=lr*2
        elif a=='r':
            lr=lr/2
#fig, axs = plt.subplots(2, 2)
#axs[0, 0].set_title('Real')
#axs[0, 0].scatter(x, y, c=tf_target.numpy(),cmap='rainbow')
#axs[0, 0].plt.plot(xl,yl)
#axs[0, 0].axis(
#                (x_min,x_max,
#                 x_min,x_max)
#                )
#                
#axs[0, 1].set_title('Pred [0,0]')
#axs[0, 1].scatter(x, y, c=model(new_sample).numpy(),cmap='rainbow')
#axs[0, 1].plt.plot(xl,yl)
#axs[0, 1].axis(
#                (x_min,x_max,
#                 x_min,x_max)
#                )
#############################################################
##------------   Option 1: Non-persistent GradientTape -----#
#############################################################
#lr=0.1
#with tf.GradientTape() as t:
#    current_loss=loss(tf_target, model(new_sample))
#    with t.stop_recording():
#      # The gradient computation below is not traced, saving memory.
#      grads = t.gradient(current_loss, model.variables)   
#current_weights=model.get_weights()
#grad=[x.numpy() for x in grads]
#new_weights=np.add(
#                    current_weights,
#                    np.multiply(grad,-lr)
#                    )
#model.set_weights(new_weights)
#


