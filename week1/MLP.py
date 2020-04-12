import tensorflow as tf
class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self,input_dim,output_dim):
        super(MyDenseLayer,self).__init__()
        #Initialize wieghts and bias
        self.W=self.add_weight([input_dim,output_dim])
        self.b=self.add_weight([1,output_dim])
        
        def call(self,inputs):
            #Forward propagate the inputs
            z=tf.matmul(inputs,self.W)+self.b
            #Feed trhough a non-linear activation
            output = tf.math.sigmoid(z)
            return output
        
def compute_loss(weights):
    loss=0
    y=None
    inputs=None
    pred=None
    
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y,predicted))
    return loss
    
            
if __name__=="__main__":
    layer = tf.keras.layers.Dense(units=2) # Creates a layer of 2 units
    
    """
        TF Sequential Models: Is an idea of composing a NN using a sequence of layers,
        you define your layers as a sequence from input layer to output layer
        
    """
    # m: #inputs
    # n: #hidden nodes
    # z: #outputs
    
    m=3
    model = tf.keras.Sequential([
                tf.keras.layers.Dense(m),
                tf.keras.layers.Dense(2)
                
            ]) #Model with n input layers and 2 output layers
    
    #Deep NN use several layers one top of each ohter
    
    n1=3
    n2=4
    n3=3
    deep_model = tf.keras.Sequential([
                tf.keras.layers.Dense(n1),
                tf.keras.layers.Dense(n2),
                tf.keras.layers.Dense(n3),
                tf.keras.layers.Dense(2)
                
            ]) #Model with n input layers and 2 output layers
    
    #Initialize weights as 
    weights= tf.Variable([tf.random.normal()])
    
    lr=0.1 #learning_rate
    #Define y and predicted!
    y=None              #ToDo
    predicted=None      #ToDo
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y,predicted))
    if False: #I really don't want to enter an infinite loop right now
        while True:
            with tf.GradientTape() as g:
                loss=compute_loss(weights)
                gradient=g.gradient(loss,weights)
            weights=weights -lr * gradient
            
            #Gradient Descent Optimization implementation in TF
#             tf.keras.optimizers.SGD       # SGD
#             tf.keras.optimizers.Adam      # Adam
#             tf.keras.optimizers.Adadelta  # Adadelta
#             tf.keras.optimizers.Adagrad   # Adagrad
#             tf.keras.optimizers.RMSProp   # RMSProp
            