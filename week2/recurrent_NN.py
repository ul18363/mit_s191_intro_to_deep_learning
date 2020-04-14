import tensorflow as tf

class MyRNNCell(tf.keras.layers.Layer):
	"""
	Manually created class representing a Recurrent Neural Network
	It consists of:
		* the input weights that connects an input vector "x" with a hidden layer "h"
		* the hidden weights that connects the hidden layer h with itself (Fully connected it seems)
		* the output weights that produce an output vector when multiplied by the hidden layer.

		A similar out of the box analogue would be using: tf. keras.layers.SimpleRNN(rnn_units)
	"""
	def __init(self,rnn_untis,input_dim,output_dim):
		super(MyRNNCell,self).__init__()

		# Initialize weight matrices
		self.W_xh=self.add_weight([rnn_untis, input_dim])
		self.W_hh=self.add_weight([rnn_untis, rnn_untis])
		self.W_hy=self.add_weight([output_dim, rnn_untis])

		#Initialize hidden state to zeros
		self.h =tf.zeros([rnn_untis,1])

	def call(self,x):
		# Update the hidden state
		self.h = tf.math.tanh(self.W_hh * self.h + self.W_xh * x)
		# Compute the output
		output = self.W_hy * self.h

		#Return the current ouput and hiddent state
		return output,self.h

if __name__=='__main__':
